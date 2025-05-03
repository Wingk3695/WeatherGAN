import os
import requests
import sys
import copy
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig

p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd

from attention_inject.DepthInjectedUNet import UNetWithDepth
from attention_inject.DepthAttnProcessor import DepthAttnProcessor
from diffusers.models.attention import Attention  # 注意确保引入正确的Attention类


class Pix2Pix_Turbo(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints",
                 lora_rank_unet=8, lora_rank_vae=4, depth_feat_dim=768):
        super().__init__()
        # --- 文本编码 ---
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()

        # --- VAE初始化及跳跃连接 ---
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, 1, bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, 1, bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, 1, bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, 1, bias=False).cuda()
        vae.decoder.ignore_skip = False

        # --- UNetWithDepth ---
        unet = UNetWithDepth.from_pretrained("stabilityai/sd-turbo", subfolder="unet")

        # --- 顶层统一定义depth相关线性层 ---
        self.depth_to_k_layers = torch.nn.ModuleDict()
        self.depth_to_v_layers = torch.nn.ModuleDict()
        for name, module in unet.named_modules():
            if isinstance(module, Attention):
                key = name.replace('.', '_')  # 替换点号
                embed_dim = module.inner_dim
                self.depth_to_k_layers[key] = torch.nn.Linear(depth_feat_dim, embed_dim)
                self.depth_to_v_layers[key] = torch.nn.Linear(depth_feat_dim, embed_dim)

        # --- 替换processor，传入引用的depth线性层 ---
        new_processors = {}
        for k, v in unet.attn_processors.items():
            k_key = k
            if k_key.endswith('.processor'):
                k_key = k_key[:-len('.processor')]
            k_key = k_key.replace('.', '_')
            # 这里k_key是depth_to_k_layers的key
            new_processors[k] = DepthAttnProcessor(
                lambda_d=5.0,
                gamma_d=1.0,
                to_k_depth=self.depth_to_k_layers[k_key],
                to_v_depth=self.depth_to_v_layers[k_key]
            )
        unet.set_attn_processor(new_processors)

        # --- 加载预训练权重和LoRA，如果有的话 ---
        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)

            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                                  "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                                  "to_k", "to_q", "to_v", "to_out.0"]
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian", target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

            target_modules_unet = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                                  "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"]
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian", target_modules=target_modules_unet)
            unet.add_adapter(unet_lora_config)

            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae
            self.target_modules_unet = target_modules_unet

        # unet.enable_xformers_memory_efficient_attention()
        unet.to("cuda")
        vae.to("cuda")
        self.unet = unet
        self.vae = vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)


    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        for k in self.depth_to_k_layers:
            self.depth_to_k_layers[k].eval()
            for p in self.depth_to_k_layers[k].parameters():
                p.requires_grad = False
        for k in self.depth_to_v_layers:
            self.depth_to_v_layers[k].eval()
            for p in self.depth_to_v_layers[k].parameters():
                p.requires_grad = False


    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

        for k in self.depth_to_k_layers:
            self.depth_to_k_layers[k].train()
            for p in self.depth_to_k_layers[k].parameters():
                p.requires_grad = True
        for k in self.depth_to_v_layers:
            self.depth_to_v_layers[k].train()
            for p in self.depth_to_v_layers[k].parameters():
                p.requires_grad = True


    def forward(self, c_t, prompt=None, prompt_tokens=None, 
                depth_feat=None, depth_mask=None,
                deterministic=True, r=1.0, noise_map=None):
        assert (prompt is None) != (prompt_tokens is None), "prompt and prompt_tokens互斥"
        if prompt is not None:
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

        if deterministic:
            model_pred = self.unet(
                encoded_control,
                self.timesteps,
                encoder_hidden_states=caption_enc,
                depth_feat=depth_feat,
                depth_mask=depth_mask,
            ).sample
            x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
            x_denoised = x_denoised.to(model_pred.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        else:
            self.unet.set_adapters(["default"], weights=[r])
            set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])
            unet_input = encoded_control * r + noise_map * (1 - r)
            self.unet.conv_in.r = r
            unet_output = self.unet(
                unet_input,
                self.timesteps,
                encoder_hidden_states=caption_enc,
                depth_feat=depth_feat,
                depth_mask=depth_mask,
            ).sample
            self.unet.conv_in.r = None
            x_denoised = self.sched.step(unet_output, self.timesteps, unet_input, return_dict=True).prev_sample
            x_denoised = x_denoised.to(unet_output.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            self.vae.decoder.gamma = r
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image


    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae

        # 这里扩展保存depth相关线性层参数（它们统一注册在顶层模型）
        sd["state_dict_unet"] = {
            k: v for k, v in self.unet.state_dict().items()
            if "lora" in k or "conv_in" in k or k.startswith("depth_to_k_layers.") or k.startswith("depth_to_v_layers.")
        }
        sd["state_dict_vae"] = {
            k: v for k, v in self.vae.state_dict().items()
            if "lora" in k or "skip" in k
        }
        torch.save(sd, outf)
