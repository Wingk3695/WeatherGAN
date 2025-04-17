import json
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random

# 1. 数据加载与预处理
def load_data(json_path, num_original=100, num_foggy=300):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    original_images = set()
    foggy_images = []
    labels = []
    
    for item in data:
        original_images.add(item['conditioning_image'])
        foggy_images.append(item['image'])
        labels.append(item['text'].split(' ')[1])
    
    # 随机采样原图和雾图
    original_list = list(original_images)
    sampled_original = random.sample(original_list, min(num_original, len(original_list)))
    sampled_foggy = random.sample(foggy_images, min(num_foggy, len(foggy_images)))
    
    # 合并采样后的图像和标签
    all_images = sampled_original + sampled_foggy
    all_labels = ['original'] * len(sampled_original) + labels[:len(sampled_foggy)]
    
    return all_images, all_labels



# 2. 特征提取
def extract_features(image_paths):
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    features = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            img = preprocess(img).unsqueeze(0)
            if torch.cuda.is_available():
                img = img.cuda()
            with torch.no_grad():
                feat = model(img).squeeze().cpu().numpy()
            features.append(feat)
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            continue
    return np.array(features)

# 3. 可视化
def plot_tsne(embeddings, labels):
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        idxs = np.where(np.array(labels) == label)[0]
        plt.scatter(embeddings[idxs, 0], embeddings[idxs, 1], 
                    c=[color], label=label, alpha=0.7, s=40)
    
    plt.title('t-SNE Visualization of Image Distributions', fontsize=14)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(title='Image Type', title_fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    plt.savefig(fname='tsne_visualization.png', dpi=300, bbox_inches='tight')

# 主流程
if __name__ == "__main__":
    # 加载数据（修改为你的JSON路径）
    image_paths, labels = load_data('/home/custom_users/wangkang/git/dataset/fog_dataset.json')
    
    # 提取特征
    print("Extracting features...")
    features = extract_features(image_paths)
    
    # t-SNE降维
    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)
    embeddings = tsne.fit_transform(features)
    
    # 可视化
    plot_tsne(embeddings, labels)