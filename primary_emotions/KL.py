import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr  # 用于计算KL散度

# 基本设置
base_save_dir = './rep_readers'  # 保存方向向量的路径
save_dir = './kl_divergence_plots'  # 保存KL散度图的路径
os.makedirs(save_dir, exist_ok=True)

# 模型文件夹（假设模型文件夹的名称为 "model_0", "model_1", ..., "model_14"）
model_folders = [f'model_{i}' for i in range(15)]

# 情感标签和颜色设置
emotions = ['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']

def load_directions_for_emotion_layer(emotion, layer):
    """加载所有模型中特定情感和特定层的方向向量"""
    directions = []
    
    for model_name in model_folders:
        model_path = os.path.join(base_save_dir, model_name)
        rep_reader_path = os.path.join(model_path, f'rep_reader_{emotion}.pkl')
        
        if os.path.exists(rep_reader_path):
            with open(rep_reader_path, 'rb') as f:
                rep_reader = pickle.load(f)

                if hasattr(rep_reader, 'directions') and layer in rep_reader.directions:
                    direction = rep_reader.directions[layer]
                    directions.append(direction.flatten())  # 展平方向向量
                else:
                    print(f"Layer {layer} not found in {model_name} for {emotion}")
        else:
            print(f"rep_reader file for {emotion} not found in {model_name}")
    
    return np.array(directions) if directions else None

def kl_divergence(p, q):
    """计算两个概率分布p和q的KL散度"""
    p = np.clip(p, 1e-10, 1)  # 防止log(0)
    q = np.clip(q, 1e-10, 1)
    return np.sum(rel_entr(p, q))

def compute_kl_divergence_for_layer(directions):
    """计算模型之间方向向量的KL散度"""
    n_models = directions.shape[0]
    kl_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                kl_matrix[i, j] = kl_divergence(directions[i], directions[j])
    
    return kl_matrix

def plot_kl_divergence(kl_div_matrix, layer, emotion):
    """绘制每一层的模型之间的KL散度变化图"""
    plt.figure(figsize=(10, 8))
    plt.imshow(kl_div_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label="KL Divergence")
    plt.title(f"KL Divergence for {emotion}, Layer {layer}")
    plt.xlabel("Model Index")
    plt.ylabel("Model Index")

    # 保存图像
    save_path = os.path.join(save_dir, f'{emotion}_layer_{layer}_kl_divergence.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved KL divergence plot for {emotion}, Layer {layer} at {save_path}")
    plt.close()

# 遍历所有情感，计算每个情感下的所有层的模型间的KL散度
for emotion in emotions:
    print(f"Processing KL Divergence for {emotion}...")
    
    for layer in range(-1, -32, -1):  # 从Layer -1 到Layer -31
        # 加载所有模型中该情感的该层的方向向量
        directions = load_directions_for_emotion_layer(emotion, layer)
        
        if directions is not None:
            # 将方向向量归一化为概率分布
            directions = np.abs(directions) / np.sum(np.abs(directions), axis=1, keepdims=True)
            
            # 计算方向向量之间的KL散度
            kl_div_matrix = compute_kl_divergence_for_layer(directions)
            
            # 绘制并保存KL散度图
            plot_kl_divergence(kl_div_matrix, layer, emotion)
        else:
            print(f"No directions found for {emotion}, Layer {layer}")
