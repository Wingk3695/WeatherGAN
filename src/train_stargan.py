import os
import gc
import lpips
import clip
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats

from img2img_turbo.pix2pix_turbo import Pix2Pix_Turbo
from training_utils import parse_args_star_training
from StarDataset import UnpairedStarDataset


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    # 初始化网络
    if args.pretrained_model_name_or_path == "stabilityai/sd-turbo":
        net_pix2pix = Pix2Pix_Turbo(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae)
        net_pix2pix.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_pix2pix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_pix2pix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gan_disc_type == "vagan_clip":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_clip, _ = clip.load("ViT-B/32", device="cuda")
    net_clip.requires_grad_(False)
    net_clip.eval()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # 优化器和学习率调度器，与你之前逻辑类似
    layers_to_opt = []
    for n, _p in net_pix2pix.unet.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)
    layers_to_opt += list(net_pix2pix.unet.conv_in.parameters())
    for n, _p in net_pix2pix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            layers_to_opt.append(_p)
    layers_to_opt += list(net_pix2pix.vae.decoder.skip_conv_1.parameters()) + \
                     list(net_pix2pix.vae.decoder.skip_conv_2.parameters()) + \
                     list(net_pix2pix.vae.decoder.skip_conv_3.parameters()) + \
                     list(net_pix2pix.vae.decoder.skip_conv_4.parameters())

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                 num_training_steps=args.max_train_steps * accelerator.num_processes,
                                 num_cycles=args.lr_num_cycles, power=args.lr_power)

    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=args.learning_rate,
                                       betas=(args.adam_beta1, args.adam_beta2),
                                       weight_decay=args.adam_weight_decay,
                                       eps=args.adam_epsilon)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
                                      num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                      num_training_steps=args.max_train_steps * accelerator.num_processes,
                                      num_cycles=args.lr_num_cycles, power=args.lr_power)

    # 使用你的非配对数据集
    dataset_train = UnpairedStarDataset(dataset_folder=args.dataset_folder,
                                       image_prep=args.train_image_prep,
                                       split="train",
                                       tokenizer=net_pix2pix.tokenizer)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size,
                                           shuffle=True, num_workers=args.dataloader_num_workers)

    # 验证集可参考之前代码
    dataset_val = UnpairedStarDataset(dataset_folder=args.dataset_folder,
                                     image_prep=args.test_image_prep,
                                     split="test",
                                     tokenizer=net_pix2pix.tokenizer)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # 准备模型与优化器等
    net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc)
    net_clip, net_lpips = accelerator.prepare(net_clip, net_lpips)

    t_clip_renorm = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    net_pix2pix.to(accelerator.device, dtype=weight_dtype)
    net_disc.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_clip.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps),
                        initial=0,
                        desc="Steps",
                        disable=not accelerator.is_local_main_process)

    # 关闭判别器的效率注意力
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    global_step = 0

    # 这里和原代码相比少了文件夹FID的计算

    # 主训练循环
    for epoch in range(args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            with accelerator.accumulate(net_pix2pix, net_disc):
                # 1) 读取数据
                x_src = batch["src_image"].to(device=accelerator.device, dtype=weight_dtype)
                c_src = batch["src_token_ids"].to(device=accelerator.device)
                x_tgt = batch["tgt_image"].to(device=accelerator.device, dtype=weight_dtype)
                c_tgt = batch["tgt_token_ids"].to(device=accelerator.device)

                B = x_src.shape[0]

                # ====== 正向流程：src -> tgt ======
                x_tgt_pred = net_pix2pix(x_src, prompt_tokens=c_tgt, deterministic=True)       # 源图生成目标域
                x_src_rec = net_pix2pix(x_tgt_pred, prompt_tokens=c_src, deterministic=True)  # 循环回源
                x_src_idt = net_pix2pix(x_src, prompt_tokens=c_src, deterministic=True)       # 身份映射

                # ====== 反向流程：tgt -> src ======
                # x_src_pred = net_pix2pix(x_tgt, prompt_tokens=c_src, deterministic=True)       # 目标图生成源域
                # x_tgt_rec = net_pix2pix(x_src_pred, prompt_tokens=c_tgt, deterministic=True)  # 循环回目标
                # x_tgt_idt = net_pix2pix(x_tgt, prompt_tokens=c_tgt, deterministic=True)       # 身份映射

                # ====== 损失计算 ======
                loss = 0.0

                # 对抗损失 - 生成器
                loss_gan_src2tgt = net_disc(x_tgt_pred, for_G=True).mean() * args.lambda_gan
                # loss_gan_tgt2src = net_disc(x_src_pred, for_G=True).mean() * args.lambda_gan
                loss_gan = loss_gan_src2tgt # + loss_gan_tgt2src
                loss += loss_gan

                # CLIP风格相似度损失
                loss_clipsim = torch.tensor(0., device=accelerator.device)
                if args.lambda_clipsim > 0:
                    # src->tgt方向CLIP损失
                    x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                    x_tgt_pred_renorm = F.interpolate(x_tgt_pred_renorm, (224, 224),
                                                    mode="bilinear",
                                                    align_corners=False)
                    clipsim_src2tgt, _ = net_clip(x_tgt_pred_renorm, c_tgt)
                    loss_clipsim_src2tgt = (1 - clipsim_src2tgt.mean() / 100) * args.lambda_clipsim

                    # tgt->src方向CLIP损失
                    # x_src_pred_renorm = t_clip_renorm(x_src_pred * 0.5 + 0.5)
                    # x_src_pred_renorm = F.interpolate(x_src_pred_renorm, (224, 224),
                    #                                 mode="bilinear",
                    #                                 align_corners=False)
                    # clipsim_tgt2src, _ = net_clip(x_src_pred_renorm, c_src)
                    # loss_clipsim_tgt2src = (1 - clipsim_tgt2src.mean() / 100) * args.lambda_clipsim

                    loss_clipsim = loss_clipsim_src2tgt #+ loss_clipsim_tgt2src
                    loss += loss_clipsim

                # 循环一致性损失
                loss_cyc_src2tgt_l1 = F.l1_loss(x_src_rec, x_src) * args.lambda_l1
                loss_cyc_src2tgt_lpips = net_lpips(x_src_rec, x_src).mean() * args.lambda_lpips
                loss_cyc_src2tgt = loss_cyc_src2tgt_l1 + loss_cyc_src2tgt_lpips
                loss_cyc = loss_cyc_src2tgt * args.lambda_cycle
                loss += loss_cyc

                # 身份映射损失
                loss_idt_src_l1 = F.l1_loss(x_src_idt, x_src) * args.lambda_l1
                loss_idt_src_lpips = net_lpips(x_src_idt, x_src).mean() * args.lambda_lpips
                loss_idt_src = loss_idt_src_l1 + loss_idt_src_lpips
                loss_idt = loss_idt_src * args.lambda_idt
                loss += loss_idt

                # 反向传播并优化生成器参数
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # 判别器训练，真实图像为真，生成图像为假

                # 判别器真实图像loss
                lossD_real_src = net_disc(x_src, for_real=True).mean() * args.lambda_gan
                lossD_real_tgt = net_disc(x_tgt, for_real=True).mean() * args.lambda_gan
                lossD_real = lossD_real_src + lossD_real_tgt

                accelerator.backward(lossD_real)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)

                # 判别器生成图像loss
                lossD_fake_src2tgt = net_disc(x_tgt_pred.detach(), for_real=False).mean() * args.lambda_gan
                # lossD_fake_tgt2src = net_disc(x_src_pred.detach(), for_real=False).mean() * args.lambda_gan
                lossD_fake = lossD_fake_src2tgt # + lossD_fake_tgt2src

                accelerator.backward(lossD_fake)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                optimizer_disc.zero_grad(set_to_none=True)

            # 日志及进度条更新
                if accelerator.is_main_process:
                    logs = {
                        "loss_gan": loss_gan.detach().item(),
                        "loss_clipsim": loss_clipsim.detach().item(),

                        # 循环一致性各部分
                        "loss_cyc_l1": loss_cyc_src2tgt_l1.detach().item(),
                        "loss_cyc_lpips": loss_cyc_src2tgt_lpips.detach().item(),
                        "loss_cyc_total": loss_cyc.detach().item(),

                        # 身份映射各部分
                        "loss_idt_l1": loss_idt_src_l1.detach().item(),
                        "loss_idt_lpips": loss_idt_src_lpips.detach().item(),
                        "loss_idt_total": loss_idt.detach().item(),

                        "lossD_real": lossD_real.detach().item(),
                        "lossD_fake": lossD_fake.detach().item(),
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                    # 可视化示例图
                    if global_step % args.viz_freq == 1:
                        log_dict = {
                            "train/source": [wandb.Image(
                                x_src[idx].float().detach().cpu(),
                                caption=f"idx={idx}, caption={batch['src_caption'][idx]}"
                            ) for idx in range(B)],

                            "train/target": [wandb.Image(
                                x_tgt[idx].float().detach().cpu(),
                                caption=f"idx={idx}, caption={batch['tgt_caption'][idx]}"
                            ) for idx in range(B)],

                            "train/src2tgt_output": [wandb.Image(
                                x_tgt_pred[idx].float().detach().cpu(),
                                caption=f"idx={idx}, caption={batch['tgt_caption'][idx]}"
                            ) for idx in range(B)],

                            "train/src2tgt_cycle": [wandb.Image(
                                x_src_rec[idx].float().detach().cpu(),
                                caption=f"idx={idx}, caption={batch['src_caption'][idx]}"
                            ) for idx in range(B)],

                            "train/src_identity": [wandb.Image(
                                x_src_idt[idx].float().detach().cpu(),
                                caption=f"idx={idx}, caption={batch['src_caption'][idx]}"
                            ) for idx in range(B)],
                            # "train/tgt2src_output": [wandb.Image(x_src_pred[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            # "train/tgt2src_cycle": [wandb.Image(x_tgt_rec[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            # "train/tgt_identity": [wandb.Image(x_tgt_idt[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]

                    # 定期保存模型
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_pix2pix).save_model(outf)

                    gc.collect()
                    torch.cuda.empty_cache()
                accelerator.log(logs, step=global_step)



if __name__ == "__main__":
    args = parse_args_star_training()
    main(args)
