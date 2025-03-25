from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
# model_name = "/storage/lb/checkpoints/Qwen/Qwen2-0.5B-Instruct"
# model_name = "/storage/lb/checkpoints/Qwen/Qwen2.5-7B-Instruct"
model_name = "/storage/lb/checkpoints/Qwen/Qwen2.5-3B-Instruct"
# model_name = "/storage/lb/checkpoints/Qwen/Qwen2.5-0.5B-Instruct"
# model_name = "/storage/lb/checkpoints/lmsys/vicuna-7b-v1.5"
# model_name = "/storage/lb/checkpoints/Qwen/Qwen2-7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f'total depth {len(model.model.layers)}')

del_n = len(model.model.layers)//2

prompt = "Give me a short introduction to large language model."
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
outputs = model(**model_inputs, output_hidden_states=True)
logits_per_layer = outputs.hidden_states[1:]
print(len(logits_per_layer), logits_per_layer[0].shape)
# import ipdb;ipdb.set_trace()
num_layers = len(logits_per_layer)
similarity_matrix = torch.zeros(num_layers, num_layers)

for i in range(num_layers):
    similarity_matrix[i, i] = 1  # 对角线上的相似度应该是 1
    for j in range(i + 1, num_layers):
        sim = F.cosine_similarity(
            logits_per_layer[i].flatten(start_dim=1),  # batch_size x (vocab_size * seq_len)
            logits_per_layer[j].flatten(start_dim=1),
            dim=-1
        ).mean().item()  # 取 batch 维度的均值
        similarity_matrix[i, j] = sim
        # similarity_matrix[j, i] = sim  # 对称矩阵

print("Layer-wise Cosine Similarity Matrix:")
print(similarity_matrix)

plt.figure(figsize=(20, 20))
plt.imshow(similarity_matrix.detach().numpy(), cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="Cosine Similarity")

# 在热图上标注数值
for i in range(num_layers):
    for j in range(i + 1, num_layers):
        plt.text(j, i, f"{similarity_matrix[i, j]*100:.1f}",
                 ha="center", va="center", color="black",
                 fontsize=10)
        

plt.xticks(range(num_layers))
plt.yticks(range(num_layers))
plt.xlabel("Layer Index")
plt.ylabel("Layer Index")
plt.title("Layer-wise Logits Cosine Similarity")
# plt.savefig('qwen2.5-7b.png')
plt.savefig('qwen2.5-3b.png')
# plt.savefig('qwen2.5-0.5b.png')
# plt.savefig('vicuna-7b.png')