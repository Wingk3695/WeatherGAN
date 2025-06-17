import gradio as gr
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T

from src.depth_model import Pix2Pix_Turbo
from attention_inject.DepthEmbedding import get_embeddings_from_convNext, get_normalized_depth_map
from ConvNeXt.convNeXt.convNext import ConvNeXt

import cv2
import urllib.request

def load_midas_model():
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    return midas, transform

midas, midas_transform = load_midas_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ConvNeXt for depth embedding
net_convNext = ConvNeXt().to(device).eval()
for p in net_convNext.parameters():
    p.requires_grad = False

def infer(input_image: Image.Image, prompt: str, model_path: str):
    try:
        # 加载推理模型（支持本地权重路径）
        net_pix2pix = Pix2Pix_Turbo(pretrained_path=model_path if model_path else None)
        net_pix2pix.eval()
        net_pix2pix.to(device)

        # 1. 获取深度图
        img = input_image.convert("RGB")
        img_midas = midas_transform(img).unsqueeze(0)
        with torch.no_grad():
            depth_pred = midas(img_midas)
            depth_pred = torch.nn.functional.interpolate(
                depth_pred.unsqueeze(1),
                size=img.size[::-1],  # (H, W)
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth_np = depth_pred.cpu().numpy()
            depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
            depth_img = (depth_norm * 255).astype(np.uint8)
            depth_pil = Image.fromarray(depth_img)

        # 2. 转为tensor
        t = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
        img_tensor = t(img).unsqueeze(0).to(device)
        depth_tensor = T.Resize((512, 512))(T.ToTensor()(depth_pil)).unsqueeze(0).to(device)

        # 3. 获取depth embedding
        with torch.no_grad():
            depth_feat = get_embeddings_from_convNext(net_convNext, depth_tensor)
            depth_mask = get_normalized_depth_map(depth_tensor)

        # 4. 文本tokenize（假设模型有tokenizer）
        if hasattr(net_pix2pix, "tokenizer"):
            prompt_tokens = net_pix2pix.tokenizer(prompt, truncation=True, padding="max_length", max_length=77, return_tensors="pt")["input_ids"].to(device)
        else:
            prompt_tokens = None

        # 5. 推理
        with torch.no_grad():
            output = net_pix2pix(
                img_tensor,
                prompt_tokens=prompt_tokens,
                depth_feat=depth_feat,
                depth_mask=depth_mask,
                deterministic=True,
            )
        output_img = output[0].detach().cpu()
        output_img = (output_img * 0.5 + 0.5).clamp(0, 1)
        output_pil = T.ToPILImage()(output_img)

        return output_pil, depth_pil, ""
    except Exception as e:
        return None, None, f"错误: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# WeatherGAN Depth-to-Image Demo")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="输入图片")
            prompt = gr.Textbox(label="文本描述", value="a foggy street")
            model_path = gr.Textbox(label="本地模型权重路径", value="", placeholder="如 ./my_model.ckpt")
            btn = gr.Button("生成")
        with gr.Column():
            output_image = gr.Image(type="pil", label="生成图片")
            depth_image = gr.Image(type="pil", label="深度图")
            error_box = gr.Textbox(label="错误信息", interactive=False)
    btn.click(fn=infer, inputs=[input_image, prompt, model_path], outputs=[output_image, depth_image, error_box])

if __name__ == "__main__":
    demo.launch()