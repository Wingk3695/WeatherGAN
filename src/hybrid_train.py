from regex import T
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from hybrid_model import Pix2Pix_Turbo
from hybrid_dataloader import HybridDataset
from torchvision import transforms
import lpips
import clip
import argparse
import os
from accelerate.utils import set_seed

def make_1step_sched():
    noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step

def train_hybrid_model(args):
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    # Initialize models
    model = Pix2Pix_Turbo(
        pretrained_name=args.pretrained_model_name_or_path,
        pretrained_path=args.pretrained_model_path,
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
    ).to(args.device)
    model.set_train()
    
    # Initialize discriminators
    # 这里应该使用 vision_aided_loss 的判别器，待修改。
    if args.gan_disc_type == "vagan_clip":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc_a = net_disc
        net_disc_b = net_disc
        net_disc_a.cv_ensemble.requires_grad_(False)
        net_disc_b.cv_ensemble.requires_grad_(False)
        net_disc_a.train()
        net_disc_b.train()
    
    # Initialize loss networks
    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_clip, _ = clip.load("ViT-B/32", device="cuda")

    net_lpips.requires_grad_(False)
    net_clip.requires_grad_(False)
    net_clip.eval()

    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_D_A = torch.optim.Adam(net_disc_a.parameters(), lr=args.lr)
    optimizer_D_B = torch.optim.Adam(net_disc_b.parameters(), lr=args.lr)

    # Load openai clip tokenizer
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer", revision=args.revision, use_fast=False,)
    noise_scheduler_1step = make_1step_sched()
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()


    # Load dataset
    dataset = HybridDataset(
        root_dir=args.dataset_folder,
        transform=args.train_image_prep,
        tokenizer=tokenizer
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # Training loop
    global_step = 0
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(total=len(data_loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for batch in data_loader:
            # 预训练阶段：仅使用配对数据
            if global_step < args.pretrain_steps:
                # Get paired data
                paired_batch = dataset[batch["idx"], "Paired", "a2b"]
                src_img = paired_batch["pixel_values_src"].to(args.device)
                tgt_img = paired_batch["pixel_values_tgt"].to(args.device)
                
                # Forward pass
                fake_img = model(src_img, "a2b")
                
                # Calculate losses
                loss_l2 = F.mse_loss(fake_img, tgt_img) * args.lambda_l2
                loss_lpips = net_lpips(fake_img, tgt_img).mean() * args.lambda_lpips
                loss_gan = net_disc_b(fake_img, for_G=True).mean() * args.lambda_gan
                
                # CLIP similarity loss
                if args.lambda_clipsim > 0:
                    t_clip_renorm = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                    fake_img_renorm = t_clip_renorm(fake_img * 0.5 + 0.5)
                    fake_img_renorm = F.interpolate(fake_img_renorm, (224, 224))
                    clipsim, _ = net_clip(fake_img_renorm, batch["input_ids"])
                    loss_clipsim = (1 - clipsim.mean() / 100) * args.lambda_clipsim
                    
                loss = loss_l2 + loss_lpips + loss_gan + loss_clipsim
            
            else:
                # Alternate between paired and unpaired training
                if global_step % 2 == 0:
                    # Get paired data
                    paired_batch = dataset[batch["idx"], "Paired", "a2b"]
                    src_img = paired_batch["pixel_values_src"].to(args.device)
                    tgt_img = paired_batch["pixel_values_tgt"].to(args.device)
                    
                    # Forward pass
                    fake_img = model(src_img, "a2b")
                    
                    # Calculate losses
                    loss_l2 = F.mse_loss(fake_img, tgt_img) * args.lambda_l2
                    loss_lpips = net_lpips(fake_img, tgt_img).mean() * args.lambda_lpips
                    loss_gan = net_disc_b(fake_img, for_G=True).mean() * args.lambda_gan
                    
                    # CLIP similarity loss
                    if args.lambda_clipsim > 0:
                        t_clip_renorm = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                        fake_img_renorm = t_clip_renorm(fake_img * 0.5 + 0.5)
                        fake_img_renorm = F.interpolate(fake_img_renorm, (224, 224))
                        clipsim, _ = net_clip(fake_img_renorm, batch["input_ids"])
                        loss_clipsim = (1 - clipsim.mean() / 100) * args.lambda_clipsim
                        
                    loss = loss_l2 + loss_lpips + loss_gan + loss_clipsim
                    
                else:
                    # Get unpaired data
                    unpaired_batch = dataset[batch["idx"], "Unpaired", "a2b"]
                    src_img = unpaired_batch["pixel_values_src"].to(args.device)
                    tgt_img = unpaired_batch["pixel_values_tgt"].to(args.device)
                    
                    # Forward cycle
                    fake_B = model(src_img, "a2b")
                    rec_A = model(fake_B, "b2a")
                    
                    # Backward cycle
                    fake_A = model(tgt_img, "b2a")
                    rec_B = model(fake_A, "a2b")
                    
                    # Calculate cycle consistency loss
                    loss_cycle_A = F.l1_loss(rec_A, src_img) * args.lambda_cycle
                    loss_cycle_B = F.l1_loss(rec_B, tgt_img) * args.lambda_cycle
                    
                    # GAN loss
                    loss_G_A = net_disc_a(fake_A, for_G=True).mean() * args.lambda_gan
                    loss_G_B = net_disc_b(fake_B, for_G=True).mean() * args.lambda_gan
                    
                    # Identity loss (optional)
                    if args.lambda_identity > 0:
                        loss_idt_A = F.l1_loss(model(tgt_img, "b2a"), tgt_img) * args.lambda_identity
                        loss_idt_B = F.l1_loss(model(src_img, "a2b"), src_img) * args.lambda_identity
                        loss = loss_cycle_A + loss_cycle_B + loss_G_A + loss_G_B + loss_idt_A + loss_idt_B
                    else:
                        loss = loss_cycle_A + loss_cycle_B + loss_G_A + loss_G_B

            # Update generator
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            # Update discriminators
            if global_step % args.d_reg_every == 0:
                optimizer_D_A.zero_grad()
                optimizer_D_B.zero_grad()
                
                # Discriminator A
                loss_D_A_real = net_disc_a(src_img, for_real=True).mean()
                loss_D_A_fake = net_disc_a(fake_A.detach(), for_real=False).mean()
                loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5 * args.lambda_gan
                
                # Discriminator B
                loss_D_B_real = net_disc_b(tgt_img, for_real=True).mean()
                loss_D_B_fake = net_disc_b(fake_B.detach(), for_real=False).mean()
                loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5 * args.lambda_gan
                
                accelerator.backward(loss_D_A + loss_D_B)
                optimizer_D_A.step()
                optimizer_D_B.step()

            # Update progress
            progress_bar.update(1)
            global_step += 1

            # Logging
            if global_step % args.logging_steps == 0:
                logs = {
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "step": global_step,
                    "epoch": epoch,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs)

        progress_bar.close()

        # Save checkpoint
        if epoch % args.save_epochs == 0:
            accelerator.save_state(f"{args.output_dir}/checkpoint-{epoch}")

def parse_args():
    parser = argparse.ArgumentParser()
    # Training strategy
    parser.add_argument("--pretrain_steps", type=int, default=10000)
    parser.add_argument("--num_epochs", type=int, default=100)
    
    # Loss weights (combining both pix2pix and cyclegan weights)
    parser.add_argument("--lambda_l2", type=float, default=1.0)
    parser.add_argument("--lambda_lpips", type=float, default=5.0)
    parser.add_argument("--lambda_gan", type=float, default=0.5)
    parser.add_argument("--lambda_clipsim", type=float, default=5.0)
    parser.add_argument("--lambda_cycle", type=float, default=10.0)
    parser.add_argument("--lambda_cycle_lpips", type=float, default=10.0)
    parser.add_argument("--lambda_identity", type=float, default=0.5)
    parser.add_argument("--lambda_idt_lpips", type=float, default=1.0)
    
    # Model configuration
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/sd-turbo")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=128, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    
    # Training details
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    
    # Dataset options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_image_prep", default="resize_256", type=str)
    
    # Logging and saving
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_epochs", type=int, default=5)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--seed", type=int, default=None)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )
    
    if accelerator.is_main_process:
        if args.report_to == "wandb":
            import wandb
            wandb.init(project="hybrid_img2img")
    
    # Start training
    train_hybrid_model(args)

if __name__ == "__main__":
    main()