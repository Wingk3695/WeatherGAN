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
from util import *


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
    """Normalize the depth map to the range [0, 1].
    Args:
        depth_map: depth map of shape (H, W)
    Returns:
        normalized_depth_map: normalized depth map of shape (H, W)
    """
    min_val = np.min(depth_map)
    max_val = np.max(depth_map)
    normalized_depth_map = (depth_map - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero
    return normalized_depth_map

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
