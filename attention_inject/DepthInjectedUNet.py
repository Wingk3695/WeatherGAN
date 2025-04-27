from diffusers import UNet2DConditionModel

class UNetWithDepth(UNet2DConditionModel):
    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        depth_feat=None,
        depth_mask=None,
        class_labels=None,
        timestep_cond=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        added_cond_kwargs=None,
        down_block_additional_residuals=None,
        mid_block_additional_residual=None,
        down_intrablock_additional_residuals=None,
        encoder_attention_mask=None,
        return_dict=True,
    ):
        # cross_attention_kwargs是传给attention processor的参数字典
        # 先拷贝保证不影响外部
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}

        # 把深度信息加入cross_attention_kwargs里
        if depth_feat is not None:
            cross_attention_kwargs['depth_feat'] = depth_feat
        if depth_mask is not None:
            cross_attention_kwargs['depth_mask'] = depth_mask

        # 调用父类forward，传入修改后的cross_attention_kwargs
        return super().forward(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,
        )
