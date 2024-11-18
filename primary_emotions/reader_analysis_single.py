import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm  # 导入颜色映射

# 模型的基路径
base_save_dir = './rep_readers'
save_dir = './tsne_plots'  # 图片保存路径

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

# 模型文件夹（假设模型文件夹的名称为 "model_0", "model_1", ..., "model_14"）
model_folders = [f'model_{i}' for i in range(15)]

# 情感标签和颜色设置
emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]

def load_directions_for_emotion(model_path, emotion):
    """加载指定情感的所有层的方向向量"""
    emotion_directions = []
    layer_labels = []

    # 遍历所有层次
    for layer in range(-1, -32, -1):  # 从 Layer -1 到 Layer -31
        rep_reader_path = os.path.join(model_path, f'rep_reader_{emotion}.pkl')
        if os.path.exists(rep_reader_path):
            with open(rep_reader_path, 'rb') as f:
                rep_reader = pickle.load(f)

                if hasattr(rep_reader, 'directions') and layer in rep_reader.directions:
                    direction = rep_reader.directions[layer]
                    emotion_directions.append(direction)  # 添加该层的方向向量
                    layer_labels.extend([layer] * direction.shape[0])  # 添加层标签
        else:
            print(f"rep_reader file for {emotion} not found in {model_path}")

    if emotion_directions:
        # 将该情感的所有层的方向向量拼接起来
        return np.vstack(emotion_directions), layer_labels
    else:
        return None, None

def perform_tsne_and_plot_per_emotion(model_name, model_path, emotion):
    """对每个模型的每个情感的所有层进行 t-SNE 并绘制结果"""
    directions, layer_labels = load_directions_for_emotion(model_path, emotion)
    if directions is None:
        print(f"No data found for {emotion} in {model_name}")
        return

    # 设置t-SNE参数并进行降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(layer_labels) - 1))
    directions_2d = tsne.fit_transform(directions)

    # 创建颜色映射（从蓝色到红色，代表层次从低到高）
    cmap = cm.get_cmap('coolwarm')  # 使用 coolwarm 渐变
    norm = plt.Normalize(vmin=min(layer_labels), vmax=max(layer_labels))  # 正则化层次值以匹配颜色范围
    colors = cmap(norm(layer_labels))  # 将每个层次映射到相应颜色

    # 绘制 t-SNE 结果
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(directions_2d[:, 0], directions_2d[:, 1], color=colors, marker='o', label=emotion)
    
    # 添加颜色条以展示层次变化
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label="Layer (from low to high)", ax=plt.gca())  # 添加颜色条并与当前轴关联

    plt.title(f"t-SNE for {emotion} (Layer color gradient) in {model_name}")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend(loc="best")

    # 保存图像
    save_path = os.path.join(save_dir, f'{model_name}_tsne_{emotion}.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved t-SNE plot for {emotion} in {model_name} at {save_path}")

    plt.show()

# 遍历每个模型并执行 t-SNE 可视化
for model_name in model_folders:
    model_path = os.path.join(base_save_dir, model_name)
    print(f"Processing {model_name}...")

    for emotion in emotions:
        print(f"Performing t-SNE for {emotion} in {model_name}")
        perform_tsne_and_plot_per_emotion(model_name, model_path, emotion)
