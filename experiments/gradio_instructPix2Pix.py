import PIL
import torch
import gradio as gr
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# 模型加载（注意：首次运行会自动下载模型）
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, safety_checker=None
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def inference(input_image, prompt, num_inference_steps=10, image_guidance_scale=1.0):
    # 处理上传图像，确保是RGB格式
    if input_image.mode != "RGB":
        input_image = input_image.convert("RGB")
    # 生成图像
    output = pipe(
        prompt,
        image=input_image,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
    )
    return output.images[0]

# Gradio接口
with gr.Blocks() as demo:
    gr.Markdown("# Instruct Pix2Pix 图像修改演示")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="上传图像")
            prompt = gr.Textbox(
                label="修改指令",
                placeholder="例如：turn him into cyborg",
                lines=2,
            )
            num_steps = gr.Slider(
                minimum=1, maximum=50, value=10, step=1, label="推理步数 (num_inference_steps)"
            )
            guidance_scale = gr.Slider(
                minimum=0.1, maximum=5.0, value=1.0, step=0.1, label="图像引导强度 (image_guidance_scale)"
            )
            run_button = gr.Button("生成修改图像")
        with gr.Column():
            output_image = gr.Image(label="修改后图像")

    run_button.click(
        inference,
        inputs=[input_image, prompt, num_steps, guidance_scale],
        outputs=output_image,
    )

if __name__ == "__main__":
    demo.launch()
