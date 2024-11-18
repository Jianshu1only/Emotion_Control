import os
import pickle
import json

# 模型保存的基路径
base_save_dir = './rep_readers'

# 列出所有模型文件夹
model_folders = [f for f in os.listdir(base_save_dir) if os.path.isdir(os.path.join(base_save_dir, f))]

for model_folder in model_folders:
    model_path = os.path.join(base_save_dir, model_folder)
    
    print(f"Loading data from {model_folder}...")

    # 加载 results.json 文件
    results_path = os.path.join(model_path, 'results.json')
    #if os.path.exists(results_path):
    #    with open(results_path, 'r') as f:
    #        results = json.load(f)
    #        print(f"Results for {model_folder}:")
    #        print(json.dumps(results, indent=4))
    
    # 加载所有 emotion 对应的 rep_reader pkl 文件
    for emotion in ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]:
        rep_reader_path = os.path.join(model_path, f'rep_reader_{emotion}.pkl')
        if os.path.exists(rep_reader_path):
            with open(rep_reader_path, 'rb') as f:
                rep_reader = pickle.load(f)

                # 只打印 directions
                if hasattr(rep_reader, 'directions'):
                    print(f"Directions for {emotion} in {model_folder}:")
                    for layer, direction in rep_reader.directions.items():
                        print(f"Layer {layer}:")
                        print(direction)  # 打印每一层的方向向量
        else:
            print(f"rep_reader file for {emotion} not found in {model_folder}")

    print("\n" + "="*50 + "\n")

