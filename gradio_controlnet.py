import gradio as gr
from transformers import pipeline, AutoImageProcessor, UperNetForSemanticSegmentation
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import torch

# ----------- Depth ControlNet -----------
device = "cuda" if torch.cuda.is_available() else "cpu"
depth_estimator = pipeline('depth-estimation', device=0 if device == "cuda" else -1)

controlnet_depth = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe_depth = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet_depth,
    safety_checker=None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe_depth.scheduler = UniPCMultistepScheduler.from_config(pipe_depth.scheduler.config)
pipe_depth.enable_xformers_memory_efficient_attention()
pipe_depth = pipe_depth.to(device)

def infer_depth2img(input_image, prompt, steps=20, guidance=7.5):
    depth = depth_estimator(input_image)['depth']
    depth = np.array(depth)
    depth = depth[:, :, None]
    depth = np.concatenate([depth, depth, depth], axis=2)
    depth = Image.fromarray(depth)
    result = pipe_depth(prompt, depth, num_inference_steps=steps, guidance_scale=guidance).images[0]
    return result

# ----------- Segmentation ControlNet -----------
palette = np.asarray([
    [0, 0, 0],
    [120, 120, 120],
    [180, 120, 120],
    [6, 230, 230],
    [80, 50, 50],
    [4, 200, 3],
    [120, 120, 80],
    [140, 140, 140],
    [204, 5, 255],
    [230, 230, 230],
    [4, 250, 7],
    [224, 5, 255],
    [235, 255, 7],
    [150, 5, 61],
    [120, 120, 70],
    [8, 255, 51],
    [255, 6, 82],
    [143, 255, 140],
    [204, 255, 4],
    [255, 51, 7],
    [204, 70, 3],
    [0, 102, 200],
    [61, 230, 250],
    [255, 6, 51],
    [11, 102, 255],
    [255, 7, 71],
    [255, 9, 224],
    [9, 7, 230],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [7, 255, 224],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [255, 122, 8],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
    [31, 255, 0],
    [255, 31, 0],
    [255, 224, 0],
    [153, 255, 0],
    [0, 0, 255],
    [255, 71, 0],
    [0, 235, 255],
    [0, 173, 255],
    [31, 0, 255],
    [11, 200, 200],
    [255, 82, 0],
    [0, 255, 245],
    [0, 61, 255],
    [0, 255, 112],
    [0, 255, 133],
    [255, 0, 0],
    [255, 163, 0],
    [255, 102, 0],
    [194, 255, 0],
    [0, 143, 255],
    [51, 255, 0],
    [0, 82, 255],
    [0, 255, 41],
    [0, 255, 173],
    [10, 0, 255],
    [173, 255, 0],
    [0, 255, 153],
    [255, 92, 0],
    [255, 0, 255],
    [255, 0, 245],
    [255, 0, 102],
    [255, 173, 0],
    [255, 0, 20],
    [255, 184, 184],
    [0, 31, 255],
    [0, 255, 61],
    [0, 71, 255],
    [255, 0, 204],
    [0, 255, 194],
    [0, 255, 82],
    [0, 10, 255],
    [0, 112, 255],
    [51, 0, 255],
    [0, 194, 255],
    [0, 122, 255],
    [0, 255, 163],
    [255, 153, 0],
    [0, 255, 10],
    [255, 112, 0],
    [143, 255, 0],
    [82, 0, 255],
    [163, 255, 0],
    [255, 235, 0],
    [8, 184, 170],
    [133, 0, 255],
    [0, 255, 92],
    [184, 0, 255],
    [255, 0, 31],
    [0, 184, 255],
    [0, 214, 255],
    [255, 0, 112],
    [92, 255, 0],
    [0, 224, 255],
    [112, 224, 255],
    [70, 184, 160],
    [163, 0, 255],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [255, 0, 143],
    [0, 255, 235],
    [133, 255, 0],
    [255, 0, 235],
    [245, 0, 255],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 41, 255],
    [0, 255, 204],
    [41, 0, 255],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [122, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [0, 133, 255],
    [255, 214, 0],
    [25, 194, 194],
    [102, 255, 0],
    [92, 0, 255],
])


image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small").to(device)

controlnet_seg = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe_seg = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet_seg,
    safety_checker=None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe_seg.scheduler = UniPCMultistepScheduler.from_config(pipe_seg.scheduler.config)
pipe_seg.enable_xformers_memory_efficient_attention()
pipe_seg = pipe_seg.to(device)

def infer_seg2img(input_image, prompt, steps=20, guidance=7.5):
    pixel_values = image_processor(input_image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        outputs = image_segmentor(pixel_values)
    seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[input_image.size[::-1]])[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    color_seg = color_seg.astype(np.uint8)
    seg_image = Image.fromarray(color_seg)
    result = pipe_seg(prompt, seg_image, num_inference_steps=steps, guidance_scale=guidance).images[0]
    return result, seg_image

with gr.Blocks() as demo:
    gr.Markdown("# ControlNet Depth2Image & Seg2Image Demo")
    with gr.Tabs():
        with gr.Tab("Depth2Image"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", label="上传原图")
                    prompt = gr.Textbox(label="Prompt", value="Stormtrooper's lecture")
                    steps = gr.Slider(1, 50, value=20, step=1, label="推理步数")
                    guidance = gr.Slider(1, 20, value=7.5, step=0.1, label="Guidance Scale")
                    btn = gr.Button("生成")
                with gr.Column():
                    output_image = gr.Image(type="pil", label="生成结果")
            btn.click(
                fn=infer_depth2img,
                inputs=[input_image, prompt, steps, guidance],
                outputs=output_image
            )
        with gr.Tab("Seg2Image"):
            with gr.Row():
                with gr.Column():
                    seg_input_image = gr.Image(type="pil", label="上传原图")
                    seg_prompt = gr.Textbox(label="Prompt", value="A house in Van Gogh style")
                    seg_steps = gr.Slider(1, 50, value=20, step=1, label="推理步数")
                    seg_guidance = gr.Slider(1, 20, value=7.5, step=0.1, label="Guidance Scale")
                    seg_btn = gr.Button("生成")
                with gr.Column():
                    seg_output_image = gr.Image(type="pil", label="生成结果")
                    seg_vis_image = gr.Image(type="pil", label="分割可视化")
            seg_btn.click(
                fn=infer_seg2img,
                inputs=[seg_input_image, seg_prompt, seg_steps, seg_guidance],
                outputs=[seg_output_image, seg_vis_image]
            )

if __name__ == "__main__":
    demo.launch()