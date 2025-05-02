import torch
from diffusers.models.attention_processor import AttnProcessor

class DepthAttnProcessor(AttnProcessor):
    def __init__(self, lambda_d=5.0, gamma_d=1.0):
        super().__init__()
        self.lambda_d = lambda_d  # 深度mask的权重系数
        self.gamma_d = gamma_d    # 融合深度特征的系数

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        depth_feat=None,   # 额外传入的深度特征  (B, 1, C)
        depth_mask=None,   # 额外传入的归一化深度图mask (B, H, W, 1)
        *args,
        **kwargs,
    ):
        print(f"hidden_states.shape: {hidden_states.shape}")
        print(f"encoder_hidden_states.shape: {encoder_hidden_states.shape if encoder_hidden_states is not None else None}")
        print(f"attention_mask.shape: {attention_mask.shape if attention_mask is not None else None}")
        print(f"temb.shape: {temb.shape if temb is not None else None}")
        print(f"depth_feat.shape: {depth_feat.shape if depth_feat is not None else None}")
        print(f"depth_mask.shape: {depth_mask.shape if depth_mask is not None else None}")

        # residual连接备份
        residual = hidden_states

        # 处理可能的空间归一化和维度转换
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            # flatten空间维度，变成sequence
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)  # (B, HW, C)

        # 处理encoder_hidden_states和attention_mask
        if encoder_hidden_states is None:
            # 没有encoder_hidden_states，直接用hidden_states计算自注意力
            encoder_hidden_states = hidden_states
        else:
            # 这里默认保持不变，如果有norm_cross可以调用
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # 计算Q,K,V
        query = attn.to_q(hidden_states)  # (B, seq_len, C)
        key = attn.to_k(encoder_hidden_states)  # (B, seq_len, C)
        value = attn.to_v(encoder_hidden_states)  # (B, seq_len, C)

        # 变成多头batch维度 (B * num_heads, seq_len, head_dim)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 计算QK相似度
        attn_weights = torch.bmm(query, key.transpose(-1, -2)) * attn.scale  # (B*num_heads, seq_len, seq_len)

        # 加入attention_mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # ----------- 加入深度 ------------

        if depth_mask is not None:
            # depth_mask是(B, H, W, 1)，先reshape到(B, HW, 1)
            if depth_mask.ndim == 4:
                b, h, w, _ = depth_mask.shape
                depth_mask_seq = depth_mask.view(b, h*w, 1)  # (B, HW, 1)
            else:
                depth_mask_seq = depth_mask  # (B, HW, 1) 或 (B, seq_len, 1)

            # repeat到多头batch维度
            batch_heads = attn_weights.shape[0]
            num_heads = batch_heads // batch_size
            depth_mask_expanded = depth_mask_seq.repeat_interleave(num_heads, dim=0)  # (B*num_heads, seq_len, 1)

            # attn_weights是(B*num_heads, seq_len, seq_len)，需要扩展depth_mask_expanded最后一维匹配seq_len
            # 这里我们简单加mask到所有"key"维度对应位置（即广播depth_mask_expanded到( B*num_heads, seq_len, seq_len)）
            # 一般mask的维度是对key维度加bias，故广播成 (B*num_heads, seq_len_query, seq_len_key)
            # depth_mask_expanded (B*num_heads, seq_len_key, 1)，转置为(B*num_heads, 1, seq_len_key)方便广播

            mask_bias = self.lambda_d * depth_mask_expanded.permute(0, 2, 1)  # (B*num_heads, 1, seq_len_key)
            attn_weights = attn_weights + mask_bias

        # 计算softmax注意力权重
        attention_probs = torch.softmax(attn_weights, dim=-1)

        # ----------- 基于深度特征加权融合 ------------

        if depth_feat is not None:
            # depth_feat是(B, 1, C)，扩展到(B*num_heads, 1, head_dim)
            depth_feat_exp = depth_feat.repeat_interleave(num_heads, dim=0)

            # 计算基于深度特征的注意力加权输出
            depth_out = torch.bmm(attention_probs, depth_feat_exp)  # (B*num_heads, seq_len, head_dim)

            # 转回多头格式(B, seq_len, C)
            depth_out = attn.batch_to_head_dim(depth_out)

            # 返回融合输出
            hidden_states = self.gamma_d * depth_out + residual
        else:
            # 标准用value加权
            attn_output = torch.bmm(attention_probs, value)
            attn_output = attn.batch_to_head_dim(attn_output)

            # 线性投影和dropout
            hidden_states = attn.to_out[0](attn_output)
            hidden_states = attn.to_out[1](hidden_states)

            # 加残差
            hidden_states = hidden_states + residual

        # 如果输入是4维，需要reshape回去
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # 除以rescale因子
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
