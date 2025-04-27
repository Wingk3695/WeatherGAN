import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor

class DepthAttnProcessor(AttnProcessor):
    def __init__(self, lambda_d=5.0, gamma_d=1.0, depth_feat_dim=768, head_dim=64):
        super().__init__()
        self.lambda_d = lambda_d  # 深度mask权重系数
        self.gamma_d = gamma_d    # 深度特征融合系数
        device = torch.device("cuda")
        self.depth_proj = torch.nn.Linear(depth_feat_dim, head_dim).to(device)

    def __call__(
        self,
        attn,
        hidden_states,            # \mathbf{Z}, shape (B, HW, C)
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        depth_feat=None,          # \mathbf{H}_d, shape (B, 1, C)
        depth_mask=None,          # M^d, shape (B, H, W, 1)
        *args,
        **kwargs,
    ):
        residual = hidden_states  # 保存残差

        # 如果4维，reshape成 (B, HW, C)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)  # (B, HW, C)

        # 没有encoder_hidden_states就自注意力
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        batch_size, seq_len, dim = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # 1) 计算 Q, K, V
        query = attn.to_q(hidden_states)                    # (B, seq_len, C)
        key = attn.to_k(encoder_hidden_states)             # (B, seq_len, C)
        value = attn.to_v(encoder_hidden_states)           # (B, seq_len, C)

        # 2) 多头展开
        query = attn.head_to_batch_dim(query)               # (B*num_heads, seq_len, head_dim)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 3) 经典注意力权重 QK^T / sqrt(d)
        attn_weights = torch.bmm(query, key.transpose(-1, -2)) * attn.scale  # (B*num_heads, seq_len, seq_len)

        # 4) 加入 attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        #以下是深度信息融合，贴合数学公式

        if depth_mask is not None:
            # 下采样深度mask到对应seq_len大小
            b, h, w, _ = depth_mask.shape
            target_size = int(seq_len ** 0.5)
            depth_mask = depth_mask.permute(0, 3, 1, 2)  # (B, 1, H, W)
            depth_mask_down = F.interpolate(depth_mask, size=(target_size, target_size), mode="bilinear", align_corners=False)
            depth_mask_down = depth_mask_down.permute(0, 2, 3, 1)  # (B, target_size, target_size, 1)
            depth_mask_seq = depth_mask_down.view(b, seq_len, 1)  # (B, seq_len, 1)

            # 重复到多头 batch 维度
            batch_heads = attn_weights.shape[0]
            num_heads = batch_heads // batch_size
            depth_mask_expanded = depth_mask_seq.repeat_interleave(num_heads, dim=0)  # (B*num_heads, seq_len, 1)

            # 先计算 self.lambda_d * M^d 部分
            mask_bias = self.lambda_d * depth_mask_expanded.permute(0, 2, 1)  # (B*num_heads, 1, seq_len)

            # QK^T / sqrt(d) 部分，对应 \mathbf{Z} \mathbf{H}_d^T / \sqrt{d}
            # 代码中 query shape: (B*num_heads, seq_len, head_dim)
            # depth_feat shape: (B, 1, C), 需要扩展多头，且head_dim可能和C匹配
            if depth_feat is None:
                # 如果没有depth_feat，单用depth_mask
                attn_weights = attn_weights + mask_bias
            else:
                if depth_feat.ndim == 2:  # (B, C)
                    depth_feat = depth_feat.unsqueeze(1)  # (B, 1, C)
                elif depth_feat.ndim != 3:
                    raise ValueError(f"depth_feat shape {depth_feat.shape} not supported")

                print("depth_feat shape:", depth_feat.shape)

                # 计算 Z H_d^T / sqrt(d)
                # attn_weights是 (B*num_heads, seq_len, seq_len)
                # 这里无法直接加到attn_weights，需要先算 Z H_d^T 形式的矩阵加到attn_weights中

                # 用 query 和 depth_feat_exp 计算点积（broadcast使用）
                # query: (B*num_heads, seq_len, head_dim)
                # depth_feat_exp: (B*num_heads, 1, head_dim)
                # 先转成相同维度，用 bmm 计算 (B*num_heads, seq_len, 1)
                # 注意: 需要保证 head_dim 对应 depth_feat 的C维度，这里假设一致

                # 将 depth_sim 变换成 (B*num_heads, seq_len, seq_len)，方便加到 attn_weights
                # 由于 depth_sim 是针对 key 维度的，广播复制 seq_len 次
                
                depth_feat_exp = depth_feat.repeat_interleave(num_heads, dim=0)  # (B*num_heads, 1, C)
                print("depth_feat_exp shape:", depth_feat_exp.shape)
                depth_feat_proj = self.depth_proj(depth_feat_exp)  # (B*num_heads, 1, head_dim)
                depth_sim = torch.bmm(query, depth_feat_proj.transpose(1, 2)) / (dim ** 0.5)  # (B*num_heads, seq_len, 1)
                depth_sim_expanded = depth_sim.expand(-1, -1, seq_len)  # (B*num_heads, seq_len, seq_len)

                # 最终叠加
                attn_weights = attn_weights + mask_bias + depth_sim_expanded

        # 计算注意力概率
        attention_probs = torch.softmax(attn_weights, dim=-1)

        # 计算输出，与数学中 A_d H_d 对应

        if depth_feat is not None:
            depth_feat_exp = depth_feat.repeat_interleave(num_heads, dim=0)  # (B*num_heads, 1, head_dim)
            depth_feat_exp = depth_feat_exp.expand(-1, seq_len, -1)         # (B*num_heads, seq_len, head_dim)
            depth_out = torch.bmm(attention_probs, depth_feat_exp)          # (B*num_heads, seq_len, head_dim)
            depth_out = attn.batch_to_head_dim(depth_out)                   # (B, seq_len, C)

            hidden_states = self.gamma_d * depth_out + residual
            # 参考公式 Z_d = gamma_d * A_d H_d + Z

        else:
            # 标准用value加权
            attn_output = torch.bmm(attention_probs, value)
            attn_output = attn.batch_to_head_dim(attn_output)
            hidden_states = attn.to_out[0](attn_output)
            hidden_states = attn.to_out[1](hidden_states)
            hidden_states = hidden_states + residual

        # 如果是4维输入，reshape回去
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, c, h, w)

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
