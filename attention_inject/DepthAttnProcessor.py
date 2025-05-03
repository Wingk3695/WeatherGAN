import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor

class DepthAttnProcessor(AttnProcessor):
    def __init__(self, lambda_d, gamma_d, to_k_depth, to_v_depth):
        super().__init__()
        self.lambda_d = lambda_d  # 深度attention logits缩放因子
        self.gamma_d = gamma_d    # 深度value加权因子
        self.to_k_depth = to_k_depth
        self.to_v_depth = to_v_depth

    def __call__(self, attn, hidden_states,
                 encoder_hidden_states=None,
                 attention_mask=None,
                 temb=None,
                 depth_feat=None,
                 depth_mask=None,
                 *args, **kwargs):
        # 1. 保留残差
        residual = hidden_states

        # 2. 空间归一化（如果有）
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        # 3. 预处理4D输入为序列
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, seq_len, _ = hidden_states.shape

        # 4. 准备序列长度，用于mask扩展等
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # 5. 处理attention_mask
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)

        # 6. 归一化
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # 7. 计算query
        query = attn.to_q(hidden_states)

        # 8. 准备key/value
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # 9. 计算depth key/value（如果有）
        if depth_feat is not None:
            k_depth = self.to_k_depth(depth_feat)
            v_depth = self.to_v_depth(depth_feat)
        else:
            k_depth, v_depth = None, None

        # 10. 调试打印
        print("*" * 20)
        print(f"depth_feat shape: {depth_feat.shape if depth_feat is not None else None}")
        print(f"depth_mask shape: {depth_mask.shape if depth_mask is not None else None}")
        print(f"key shape: {key.shape}, value shape: {value.shape}")
        print(f"k_depth shape: {k_depth.shape if k_depth is not None else None}, v_depth shape: {v_depth.shape if v_depth is not None else None}")
        print("*" * 20)

        # 11. reshape heads，兼容原attn接口
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        print("after head_to_batch_dim")
        print(f"query shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}")

        if k_depth is not None and v_depth is not None:
            k_depth = attn.head_to_batch_dim(k_depth)
            v_depth = attn.head_to_batch_dim(v_depth)

        batch_heads, lq, dim = query.shape
        lk = key.shape[1]

        # 12. 准备depth mask，扩展到[b*h, lq, lk]方便masked_fill
        def prepare_depth_mask(depth_mask, batch_size, heads, lq, lk):
            if depth_mask is None:
                return None
            if depth_mask.ndim == 4:
                B, H, W, C = depth_mask.shape
                depth_mask = depth_mask.view(B, H * W, C)
            if depth_mask.shape[-1] == 1:
                depth_mask = depth_mask.squeeze(-1)  # [B, lk]
            if depth_mask.shape[1] != lk:
                depth_mask = depth_mask.mean(dim=1, keepdim=True)  # 降维适配
            depth_mask = depth_mask[:, None, None, :]  # [B,1,1,lk]
            depth_mask = depth_mask.expand(-1, heads, lq, -1)  # [B, heads, lq, lk]
            depth_mask = depth_mask.reshape(batch_size * heads, lq, lk)
            return depth_mask

        depth_mask_proc = None
        heads = attn.heads
        if depth_mask is not None and k_depth is not None:
            # depth_mask一般原始是[B,H,W,1]，先调整顺序和大小
            depth_mask_permute = depth_mask.permute(0, 3, 1, 2)  # [B,1,H,W]
            cur_size = lk  # 假设lk等于空间token数，比如64*64=4096
            print(f"converting depth_mask from {depth_mask.shape} to {cur_size}x{cur_size}")
            depth_mask_down = F.interpolate(depth_mask_permute, size=(int(cur_size**0.5), int(cur_size**0.5)), mode='bilinear', align_corners=False)
            depth_mask_down = depth_mask_down.permute(0, 2, 3, 1)  # [B, H*W, 1]
            print(f"depth_mask_down shape: {depth_mask_down.shape}")
            depth_mask_proc = prepare_depth_mask(depth_mask_down, batch_size, heads, lq, lk)
            print(f"depth_mask_proc shape: {depth_mask_proc.shape}")

        # 13. 计算主attention logits  QK^T / sqrt(d)
        scale = dim ** -0.5
        attn_logits = torch.bmm(query, key.transpose(1, 2)) * scale

        # 14. 计算depth attention logits 并融合
        attn_logits_depth = None
        if k_depth is not None:
            attn_logits_depth = torch.bmm(query, k_depth.transpose(1, 2)) * scale
            if depth_mask_proc is not None:
                attn_logits_depth = attn_logits_depth.masked_fill(~depth_mask_proc, float('-1e9'))
            attn_logits = attn_logits + self.lambda_d * attn_logits_depth

        # 15. 结合标准attention_mask
        if attention_mask is not None:
            attn_logits = attn_logits.masked_fill(~attention_mask.bool(), float('-1e9'))

        # 16. 计算注意力分布
        attn_probs = torch.softmax(attn_logits, dim=-1)

        # 17. 注意力加权value
        attn_output = torch.bmm(attn_probs, value)

        # 18. 深度value加权叠加
        if k_depth is not None and v_depth is not None and attn_logits_depth is not None:
            attn_probs_depth = torch.softmax(attn_logits_depth, dim=-1)
            depth_out = torch.bmm(attn_probs_depth, v_depth)
            attn_output = attn_output + self.gamma_d * depth_out

        # 19. 恢复多头维度 [B, heads, L, head_dim]
        attn_output = attn.batch_to_head_dim(attn_output)

        # 20. 输出线性映射和dropout
        out = attn.to_out[0](attn_output)
        out = attn.to_out[1](out)

        # 21. 4D输入时reshape回去
        if input_ndim == 4:
            out = out.transpose(1, 2).reshape(batch_size, channel, height, width)

        # 22. 残差连接和缩放
        if attn.residual_connection:
            out = out + residual
        out = out / attn.rescale_output_factor

        return out
