from turtle import forward
from transformers import PretrainedConfig


class MyMinimindConfig(PretrainedConfig):
    model_type = "myminimind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import math
import torch.nn as nn
from torch.nn import init
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * x
    
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
    
def precompute_freqs(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    # 1. 初始化标准 RoPE 频率。
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ... dim-2]
    # 计算出的 freqs 就是标准的 1 / (base ** (2i / d))
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )

    if rope_scaling is not None:
        # 2. 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        # 只有当要推断的长度大于原始训练长度时，才应用缩放
        if end / orig_max > 1.0:
            # 3. 使用前文推导的公式，定义波长比例 b 到维度索引 i 的映射函数
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )

            # 4. 计算高频区和低频区的维度切分点
            # low: 不需要缩放的高频部分的最高索引
            # high: 需要完全缩放的低频部分的最低索引
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # 5. 计算混合因子 γ (Ramp)
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            # 6. 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp在0-1之间时：平滑过渡。
            freqs = freqs * (1 - ramp + ramp / factor)

    # 7. 根据目标长度 end，生成位置索引向量 t
    t = torch.arange(end, device=freqs.device)

    # 8. 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    freqs = torch.outer(t, freqs).float()

    # 9. 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

class MoeGate(nn.Module):
    def __init__(self, config: MyMinimindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.gate_proj = nn.Linear(self.gating_dim, self.n_routed_experts, bias=False)


    def forward(self, hidden_states):
        #  init parameters
        bsz, seq_len, h = hidden_states.shape
        hidden_states = torch.reshape(hidden_states, (-1, h))
        logits = self.gate_proj(hidden_states)

        if (self.scoring_func == "softmax"):
            scores = F.softmax(logits, dim=-1)
        else:
            raise NotImplementedError(f"Scoring function {self.scoring_func} not implemented")

        # compute gating scores and norm topk prob if needed
        topk_weight, topk_idx = torch.topk(scores, self.top_k, dim=-1) + 1e-20

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # compute aux loss either sequence-wise or batch-wise
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores.view(bsz, seq_len, -1)
                ce = torch.zero(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (fi * Pi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss
                


class Attention(nn.Module):
    def __init__(self, config: MyMinimindConfig):
        super().__init__()
        # set num_key_value_heads and assert
        self.num_key_value_heads = (
            config.num_attention_heads 
            if config.num_key_value_heads is None
            else config.num_key_value_heads
        )
        # set input/output size of layers
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # init q,k,v,o layers
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias = False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_local_kv_heads * self.head_dim, bias = False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_local_kv_heads * self.head_dim, bias = False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias = False
        )

        # init two dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # set flash attention
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and config.flash_attention
        )

    def forward(
            self, 
            x: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache=False,
            attention_mask: Optional[torch.Tensor] = None,
        ):
            bsz, seq_len, _ = x.shape 
            # deal with q,k,v
            xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            xq = torch.reshape(xq, (bsz, seq_len, self.n_local_heads, self.head_dim))
            xk = torch.reshape(xk, (bsz, seq_len, self.n_local_kv_heads, self.head_dim))
            xv = torch.reshape(xv, (bsz, seq_len, self.n_local_kv_heads, self.head_dim))

            # deal with position_embedding
            cos, sin = position_embeddings
            xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

            # deal with kv_cache
            if past_key_value is not None:
                xk = torch.cat([past_key_value[0], xk], dim=1)
                xv = torch.cat([past_key_value[1], xv], dim=1)
            past_kv = (xk, xv) if use_cache else None

            # alignment xq, xk, xv
            xq, xk, xv = (
                xq.transpose(1,2),
                repeat_kv(xk, self.n_rep).transpose(1,2),
                repeat_kv(xv, self.n_rep).transpose(1,2)
            )

            # use or not use flash_attention
            if (
                self.flash
                and (seq_len > 1)
                and (past_key_value is None)
                and (attention_mask is None or torch.all(attention_mask == 1))
            ):
                # use flash_attention
                output = F.scaled_dot_product_attention(
                    xq,
                    xk,
                    xv,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                )
            else:
                # not use flash_attention
                # compute scores = q @ v^t / sqrt(d_k)
                scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

                # deal with causal attention mask
                scores[:, :, :, -seq_len:] += torch.triu(
                    torch.full((seq_len, seq_len), float('-inf'), dtype=scores.dtype, device=scores.device),
                    diagonal=1 
                )

                # deal with regressive attention mask
                if attention_mask is not None:
                    extenden_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    extenden_attention_mask = (1.0 - extenden_attention_mask) * 1e-9
                    scores = scores + extenden_attention_mask

                # compute scores = scores * v
                scores = F.softmax(scores.float(), dim=-1).type_as(xq)
                scores = self.attn_dropout(scores)
                output = torch.matmul(scores, xv)
            
            output = output.transpose(1,2).reshape(bsz, seq_len, -1)
            output = self.resid_dropout(self.o_proj(output))
            return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: MyMinimindConfig):
        super().__init__()

        # set intermediate size
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        
        # init down_proj, up_proj, gate_proj
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))
    
class MyMinimindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MyMinimindConfig):
        super().__init__()
        # set input/output size
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads

        # set attentions, RMSNonms, mpl
        self.layer_id = layer_id
        
        self.before_attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = Attention(config)
        self.before_mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)
    
    def forward(
        self,
        hidden_states,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # residual connection -- attention
        res = hidden_states
        hidden_states, present_key_value = self.attention(
            self.before_attention_norm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )

        hidden_states = res + hidden_states

        # residual connection -- mlp
        hidden_states = hidden_states + self.mlp(
            self.before_mlp_norm(hidden_states),
        )

        return hidden_states, present_key_value

class MyMinimindModel(nn.Module):
    def __init__(self, config: MyMinimindConfig):
        super().__init__()

        # init paremeters
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers

        # init layers
        self.embed_layer = nn.Embedding(self.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MyMinimindBlock(i, config) for i in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # per-compute ROPE
        freq_cos, freq_sin = precompute_freqs(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        # register buffer
        self.register_buffer("freq_cos", freq_cos, persistent=False)
        self.register_buffer("freq_sin", freq_sin, persistent=False)
        
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
            **kwargs,
        ):
        # init parameters
        batch_size, seq_len = input_ids.shape
        if hasattr(past_key_values, "layers"):
            past_key_values = None   
        past_key_values = past_key_values or [None] * len(self.layers)

        # compute start positon
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        # embedding layers
        hidden_states = self.dropout(self.embed_layer(input_ids))

        # position embedding layers
        position_embeddings = (
            self.freq_cos[start_pos : start_pos + seq_len],
            self.freq_sin[start_pos : start_pos + seq_len]
        )

        # attention layers
        presents = []
            # TODO delete layer_index
        for layer_index, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value,
                use_cache,
                attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        return hidden_states, presents
    
class MyMinimindForCausalLm(PreTrainedModel, GenerationMixin):
    config_class = MyMinimindConfig

    def __init__(self, config: MyMinimindConfig):
        super().__init__(config)
        self.model = MyMinimindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.embed_layer.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **args,
    ):
        hidden_states, past_key_values = self.model(
            input_ids,
            attention_mask,
            past_key_values,
            use_cache,
            **args,
        )

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss,  
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
