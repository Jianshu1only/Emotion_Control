import streamlit as st
import torch
import pickle
import copy
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from repe import repe_pipeline_registry
from utils import primary_emotions_function_dataset

# 初始化 RePE 注册表
repe_pipeline_registry()

# 使用 Streamlit 缓存模型和分词器加载
@st.cache_resource
def load_model_and_tokenizer(model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True).eval()
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False, trust_remote_code=True)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    return model, tokenizer

# 缓存加载的模型和分词器
model_name_or_path = "LLM360/CrystalChat"
model, tokenizer = load_model_and_tokenizer(model_name_or_path)

# 加载情绪的 rep_reader 数据
emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
emotion_rep_readers = {}
for emotion in emotions:
    with open(f"rep_reader_{emotion}.pkl", "rb") as f:
        emotion_rep_readers[emotion] = pickle.load(f)

# 为每个情绪定义特定的 layer_id
emotion_layer_map = {
    "happiness": list(range(-1, -11, -1)),
    "sadness": list(range(-1, -12, -1)),
    "anger": list(range(-1, -10, -1)),
    "fear": list(range(-1, -26, -1)),
    "disgust": list(range(-1, -14, -1)),
    "surprise": list(range(-1, -10, -1))
}

# Streamlit 应用界面
st.title("Emotion Control Generation")
st.write("Please give your input, and this script will inject emotion into model's behavior")

# 用户输入
prompt = st.text_area("Input:", "You are stupid, I don't like you anymore!!")
emotion = st.selectbox("Choose emotions:", emotions)
coeff = st.slider("Coeff:", 1.0, 100.0, 80.0)
max_new_tokens = st.number_input("Max numbers:", min_value=10, max_value=512, value=256)

# 生成响应
if st.button("Response"):
    # 获取对应情绪的层
    layer_id = emotion_layer_map[emotion]

    # 深拷贝模型，以避免对原始模型的更改影响后续运行
    model_copy = copy.deepcopy(model)

    # 动态设置控制生成管道
    rep_control_pipeline = pipeline(
        "rep-control",
        model=model_copy,
        tokenizer=tokenizer,
        layers=layer_id,
        block_name="decoder_block",
        control_method="reading_vec"
    )
    inputs = [f"You are a chatbot, generate a long response. {prompt} "]
 
    # inputs = [f"[INST] <<SYS>> You are a chatbot, generate a long response. <</SYS>> [INST] {prompt} [/INST]"]
    rep_reader = emotion_rep_readers[emotion]

    # 设置激活向量
    activations = {
        layer: torch.tensor(coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer])
        .to(model_copy.device)
        .half()
        for layer in layer_id  # 使用每个情绪特定的层
    }

    # 生成基线和控制响应
    baseline_outputs = rep_control_pipeline(inputs, batch_size=1, max_new_tokens=max_new_tokens, do_sample=False)
    control_outputs = rep_control_pipeline(inputs, activations=activations, batch_size=1, max_new_tokens=max_new_tokens, do_sample=True)

    # 显示响应
    st.write("**Original output:**")
    st.write(baseline_outputs[0][0]['generated_text'].replace(inputs[0], ""))
    st.write(f"**Controlled output ({emotion.capitalize()}):**")
    st.write(control_outputs[0][0]['generated_text'].replace(inputs[0], ""))
