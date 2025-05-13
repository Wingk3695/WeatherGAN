import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor
class DepthAttnProcessor(AttnProcessor):
    def __init__(self, lambda_d, gamma_d, to_k_depth, to_v_depth, debug=False):
        super().__init__()
        self.lambda_d = lambda_d
        self.gamma_d = gamma_d
        self.to_k_depth = to_k_depth
        self.to_v_depth = to_v_depth
        self.debug = debug

    def __call__(self, attn, hidden_states,
                 encoder_hidden_states=None,
                 attention_mask=None,
                 temb=None,
                 depth_feat=None,
                 depth_mask=None,
                 *args, **kwargs):
        
        def debug_print(*args_, **kwargs_):
            if self.debug:
                print(*args_, **kwargs_)

        debug_print()
        debug_print("*" * 20)
        debug_print(f"hidden_states.shape:{hidden_states.shape}")
        debug_print(f"encoder_hidden_states.shape:{encoder_hidden_states.shape if encoder_hidden_states is not None else None}")

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, seq_len, _ = hidden_states.shape

        batch_size = hidden_states.shape[0]
        lq = hidden_states.shape[1]

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, lq, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        lk = key.shape[1]

        if depth_feat is not None:
            k_depth = self.to_k_depth(depth_feat)
            v_depth = self.to_v_depth(depth_feat)
        else:
            k_depth, v_depth = None, None

        debug_print("*" * 20)
        debug_print(f"depth_feat shape: {depth_feat.shape if depth_feat is not None else None}")
        debug_print(f"depth_mask shape: {depth_mask.shape if depth_mask is not None else None}")
        debug_print(f"key shape: {key.shape}, value shape: {value.shape}")
        debug_print(f"k_depth shape: {k_depth.shape if k_depth is not None else None}, v_depth shape: {v_depth.shape if v_depth is not None else None}")
        debug_print("*" * 20)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if k_depth is not None and v_depth is not None:
            k_depth = k_depth.unsqueeze(1)
            v_depth = v_depth.unsqueeze(1)
            k_depth = attn.head_to_batch_dim(k_depth)
            v_depth = attn.head_to_batch_dim(v_depth)

        debug_print("after head_to_batch_dim")
        debug_print(f"query shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}")
        if k_depth is not None and v_depth is not None:
            debug_print(f"k_depth shape:{k_depth.shape}, v_depth shape:{v_depth.shape}")

        scale = query.shape[2] ** -0.5

        def prepare_depth_mask(depth_mask, batch_size, heads, seq_len):
            if depth_mask is None:
                return None
            B, H, W, C = depth_mask.shape
            expected_tokens = H * W
            assert expected_tokens == seq_len, f"depth_mask spatial size {H}*{W}={expected_tokens} != seq_len={seq_len}"
            depth_mask = depth_mask.view(B, expected_tokens, C)
            depth_mask = depth_mask.unsqueeze(1).expand(-1, heads, -1, -1)
            depth_mask = depth_mask.reshape(B * heads, seq_len, C)
            return depth_mask

        depth_mask_proc = None
        heads = attn.heads

        if depth_mask is not None and k_depth is not None:
            depth_mask_permute = depth_mask.permute(0, 3, 1, 2)
            cur_size = int(lk ** 0.5)
            debug_print(f"converting depth_mask from {depth_mask.shape} to {cur_size}x{cur_size}")
            depth_mask_down = F.interpolate(depth_mask_permute,
                                           size=(cur_size, cur_size),
                                           mode='bilinear', align_corners=False)
            depth_mask_down = depth_mask_down.permute(0, 2, 3, 1)
            debug_print(f"depth_mask_down shape: {depth_mask_down.shape}")

            if cur_size * cur_size == lk:
                depth_mask_proc = prepare_depth_mask(depth_mask_down, batch_size, heads, lk)
                debug_print(f"depth_mask_proc shape: {depth_mask_proc.shape}")
            else:
                depth_mask_proc = None
                debug_print(f"Warning: depth_mask_down spatial size {cur_size*cur_size} != key length {lk}, skipping depth_mask")

        attn_logits = torch.bmm(query, key.transpose(1, 2)) * scale
        debug_print(f"attn_logits shape:{attn_logits.shape}")

        attn_logits_depth = None
        if k_depth is not None:
            attn_logits_depth = torch.bmm(query, k_depth.transpose(1, 2)) * scale
            debug_print(f"attn_logits_depth shape: {attn_logits_depth.shape}")
            if depth_mask_proc is not None:
                attn_logits_depth = attn_logits_depth + self.lambda_d * depth_mask_proc
                

        if attention_mask is not None:
            attn_logits = attn_logits.masked_fill(~attention_mask.bool(), float('-1e9'))

        attn_probs = torch.softmax(attn_logits, dim=-1)
        attn_output = torch.bmm(attn_probs, value)

        if attn_logits_depth is not None and v_depth is not None:
            attn_probs_depth = torch.softmax(attn_logits_depth, dim=1)
            v_depth_expanded = v_depth.expand(-1, lq, -1)
            depth_out = attn_probs_depth * v_depth_expanded
            attn_output = attn_output + self.gamma_d * depth_out

        attn_output = attn.batch_to_head_dim(attn_output)

        out = attn.to_out[0](attn_output)
        out = attn.to_out[1](out)

        if input_ndim == 4:
            out = out.transpose(1, 2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            out = out + residual
        out = out / attn.rescale_output_factor

        return out

