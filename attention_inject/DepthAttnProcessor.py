import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0

class DepthAttnProcessor(AttnProcessor2_0):
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
                 depth_mask=None):
        """
        attn: Attention模块实例
        hidden_states: [B, seq_len, dim] 或 [B, C, H, W]
        encoder_hidden_states: cross-attention输入
        attention_mask: 标准attention mask，形状可广播
        depth_feat: [B, seq_len_depth, depth_feat_dim]
        depth_mask: [B, seq_len_depth]，0或1 mask，广播成logits加权
        """

        cur_size = attn.inner_dim // attn.heads
        # print(f"attn_dim: {cur_attn_size}, attn.heads: {attn.heads}, attn.inner_dim: {attn.inner_dim}")

        # 1. 处理 hidden_states 维度
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            # [B, C, H, W] -> [B, H*W, C]
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, seq_len, _ = hidden_states.shape

        # 2. 计算query
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        # 3. 准备context（key, value的输入）
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        ################计算深度################

        # 4. 计算depth key/value
        if depth_feat is not None:
            k_depth = self.to_k_depth(depth_feat)
            v_depth = self.to_v_depth(depth_feat)
        else:
            k_depth = None
            v_depth = None

        print("" + "*" * 20)
        print(f"depth_feat shape: {depth_feat.shape if depth_feat is not None else None}")
        print(f"depth_mask shape: {depth_mask.shape if depth_mask is not None else None}")
        print(f"key shape: {key.shape}, value shape: {value.shape}")
        print(f"k_depth shape: {k_depth.shape if k_depth is not None else None}, v_depth shape: {v_depth.shape if v_depth is not None else None}")
        print("" + "*" * 20)

        # 5. reshape为多头格式 [B, L, H*D] -> [B, heads, L, head_dim]
        def reshape_heads(x):
            head_dim = attn.inner_dim // attn.heads
            return x.view(batch_size, -1, attn.heads, head_dim).permute(0, 2, 1, 3)

        query = reshape_heads(query)
        key = reshape_heads(key)
        value = reshape_heads(value)

        if k_depth is not None and v_depth is not None:
            k_depth = reshape_heads(k_depth)
            v_depth = reshape_heads(v_depth)

        print(f"query shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}")
        print(f"k_depth shape: {k_depth.shape if k_depth is not None else None}, v_depth shape: {v_depth.shape if v_depth is not None else None}")

        # 6. mask: [B, H, W, 1] -> [B, H*W, 1]
        def prepare_depth_mask(depth_mask, batch_size, heads, lq, lk):
            """
            depth_mask: [B, H, W, 1] 或 [B, H*W, 1], float tensor，归一化连续值
            目标：扩展为 [B*heads, lq, lk] 便于加到 depth attention logits 上

            注意：
            - lq 是query序列长度（通常是 H*W）
            - lk 是key序列长度（通常和 lq 一样，如果depth_feat是同一空间尺寸）
            """
            print("depth_mask shape:", depth_mask.shape, "numel:", depth_mask.numel())
            print("batch_size, heads, lq, lk:", batch_size, heads, lq, lk)



            if depth_mask is None:
                return None

            # 如果是4维，reshape到 [B, H*W, 1]
            if depth_mask.ndim == 4:
                B, H, W, C = depth_mask.shape
                depth_mask = depth_mask.view(B, H * W, C)

            # 确认depth_mask形状 [B, lk, 1] 或者 [B, lk]
            # 如果是 [B, lk, 1]， squeeze最后一维
            if depth_mask.shape[-1] == 1:
                depth_mask = depth_mask.squeeze(-1)  # [B, lk]

            # 扩展维度匹配attention logits形状 [B, heads, lq, lk]
            # 这里假设lq == lk == H*W
            depth_mask = depth_mask[:, None, None, :]  # [B,1,1,lk]
            depth_mask = depth_mask.expand(-1, heads, lq, -1)  # [B, heads, lq, lk]

            # reshape合并 batch 和 heads 维度
            depth_mask = depth_mask.reshape(batch_size * heads, lq, lk)  # [B*heads, lq, lk]

            return depth_mask

        depth_mask_proc = None
        if depth_mask is not None and k_depth is not None:
            # query.shape = [B, heads, lq, head_dim]
            batch_size = query.shape[0]
            heads = query.shape[1]
            lq = query.shape[2]
            lk = k_depth.shape[2]

            depth_mask = depth_mask.permute(0,3,1,2)  # [B,1,H,W]
            depth_mask_down = F.interpolate(depth_mask, size=(cur_size,cur_size), mode='bilinear', align_corners=False)
            depth_mask_down = depth_mask_down.permute(0,2,3,1)  # [B,64,64,1]

            depth_mask_proc = prepare_depth_mask(depth_mask_down, batch_size, heads, lq, lk)

        # 7. 计算原始attention logits QK^T / sqrt(d)
        scale = query.shape[-1] ** -0.5
        attn_logits = torch.matmul(query, key.transpose(-2, -1)) * scale

        # 8. 计算depth attention logits
        if k_depth is not None:
            attn_logits_depth = torch.matmul(query, k_depth.transpose(-2, -1)) * scale
            # 将depth_mask转换为logits形式，mask==0对应-1e9极小值
            if depth_mask_proc is not None:
                attn_logits_depth = attn_logits_depth.masked_fill(~depth_mask_proc, float('-1e9'))
            # 乘以lambda_d权重
            attn_logits = attn_logits + self.lambda_d * attn_logits_depth
        else:
            attn_logits_depth = None

        # 9. 结合attention_mask，mask掉不关心位置
        # if attention_mask is not None:
        #     attention_mask_proc = prepare_depth_mask(attention_mask, batch_size, heads, lq, lk)
        # if attention_mask_proc is not None:
        #     attn_logits = attn_logits.masked_fill(~attention_mask_proc, float('-1e9'))

        # 10. 计算注意力
        attn_probs = torch.softmax(attn_logits, dim=-1)
        attn_output = torch.matmul(attn_probs, value)

        # 11. 深度value的加权叠加
        if k_depth is not None and v_depth is not None:
            # 计算depth attention概率
            if attn_logits_depth is not None:
                attn_probs_depth = F.softmax(attn_logits_depth, dim=-1)
                # 计算深度value加权和
                depth_out = torch.matmul(attn_probs_depth, v_depth)
                attn_output = attn_output + self.gamma_d * depth_out

        # 12. 恢复 [B, L, H*head_dim]
        # head_dim = attn.inner_dim // attn.heads
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, query.shape[2], attn.inner_dim)

        # 13. 输出映射
        # print(type(attn.to_out))       # <class 'torch.nn.modules.container.ModuleList'>
        # print(len(attn.to_out))        # 2
        # print(attn.to_out)             # lora.Linear 和 Dropout(p=0.0, inplace=False)

        out = attn.to_out[0](attn_output)
        out = attn.to_out[1](out)


        # 14. 如果输入是4维，reshape回去
        if input_ndim == 4:
            out = out.transpose(1, 2).view(batch_size, -1, height, width)

        return out
