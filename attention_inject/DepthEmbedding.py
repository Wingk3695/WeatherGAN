# Use ConvNext blocks to embed the depth features
from gc import disable
import torch
import numpy as np
import sys
import os
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ConvNeXt.convNeXt.convNext import ConvNeXt
from einops import rearrange, repeat



config = {
    'DATA_ROOT': r'D:\GIT\WeatherGAN\Checkpoints\convnext_tiny_1k_224_ema.pth',
    'MODEL_NAME': 'convnext_tiny',
    'MODEL_TYPE': 'convnext',
    'MODEL_SIZE': 224,
    'NUM_CLASSES': 1000,
    'NUM_CHANNELS': 768,
    'NUM_LAYERS': 12,
    'NUM_HEADS': 12,
    'NUM_GROUPS': 32,
    'DROPOUT_RATE': 0.0,
}

def read_official_convnext_ckpt(ckpt_path):      
    "Read offical pretrained convnext ckpt and convert into my style" 
    print( "\n" + "*" * 20 + " load model from {}!".format(ckpt_path) + " *" * 20 + "\n")

    state_dict = torch.load(ckpt_path, map_location="cpu")
  
    return state_dict

def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False

def get_embeddings_from_convNext(model, input_image):
    """Input an image tensor and get the embeddings from convnext model.
    Args:
        model: convnext model
        input_image: image tensor of shape (B, C, H, W)
    """
    # model = model.to(device)
    # input_image = input_image.to(device)

    with torch.no_grad():
        embeddings = model(input_image)

    return embeddings

def get_normalized_depth_map(depth_map):
    """
    depth_map: Tensor, shape (B, 3, H, W)
    返回 shape: (B, H, W, 1), 归一化到[0, 1]
    """

    assert depth_map.ndim == 4, "Depth map should be a 4D tensor (B, C, H, W)"
    assert depth_map.shape[1] == 3, "Depth map should have 3 channels"
    if isinstance(depth_map, torch.Tensor):
        # 转成单通道
        depth_gray = depth_map.mean(dim=1, keepdim=True)  # (B,1,H,W)
        min_val = depth_gray.amin(dim=[1,2,3], keepdim=True)  # (B,1,1,1)
        max_val = depth_gray.amax(dim=[1,2,3], keepdim=True)  # (B,1,1,1)
        norm_depth = (depth_gray - min_val) / (max_val - min_val + 1e-8)  # (B,1,H,W)

        # 转换形状为 (B,H,W,1)，符合DepthAttnProcessor预期
        norm_depth = norm_depth.permute(0, 2, 3, 1).contiguous()

        return norm_depth

    else:
        # np.array版本，类似逻辑
        depth_gray = depth_map.mean(axis=1, keepdims=True)  # (B,1,H,W)
        min_val = depth_gray.min(axis=(2,3), keepdims=True)
        max_val = depth_gray.max(axis=(2,3), keepdims=True)
        norm_depth = (depth_gray - min_val) / (max_val - min_val + 1e-8)
        norm_depth = np.transpose(norm_depth, (0, 2, 3, 1))  # (B,H,W,1)
        return norm_depth



if __name__ == "__main__":
    # Test the functionality of the code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    convnext = ConvNeXt().to(device)

    convnext_tiny_checkpoint = read_official_convnext_ckpt(os.path.join(config['DATA_ROOT']) )
    convnext_tiny_checkpoint['model'].pop('head.weight')
    convnext_tiny_checkpoint['model'].pop('head.bias')
    convnext.load_state_dict(convnext_tiny_checkpoint['model'], strict=False)

    convnext.eval()
    disable_grads(convnext)

    random_input = torch.randn(1, 3, config['MODEL_SIZE'], config['MODEL_SIZE']).to(device)

    image_path = r'TestImage\berlin_1.png'
    input_image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 保证三通道
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, (config['MODEL_SIZE'], config['MODEL_SIZE']))
    input_image = torch.from_numpy(input_image).float()
    input_image = input_image.permute(2, 0, 1)  # (C, H, W)
    input_image = input_image.unsqueeze(0)  # (1, C, H, W)
    input_image = input_image.to(device)
    print("Input image shape:", input_image.shape)  # 应为 (1, 3, 224, 224)

    embeddings = get_embeddings_from_convNext(convnext, input_image)
    print("Embeddings shape:", embeddings.shape)  # Should be (1, 768) for convnext_tiny
    # actual output:
    # Input image shape: torch.Size([1, 3, 224, 224])
    # Embeddings shape: torch.Size([1, 768])
