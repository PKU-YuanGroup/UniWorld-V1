import torch
from torch import nn
ori = torch.load("/data/logs/moc/did_s_100kx1024_qf1x1_img0p0_uncond_zeroqf/checkpoints/0100000.pt")
tri = torch.load("/data/logs/moc/ft_did_s_100kx1024_qf1x1_img0p0_uncond_zeroqf_train_all/checkpoints/0020000.pt")
import ipdb;ipdb.set_trace()

ori_emb = ori['model']['module.y_embedder.embedding_table.weight'].cpu()
tri_emb = tri['model']['module.y_embedder.embedding_table.weight'].cpu()
print(torch.allclose(ori_emb[:1000], tri_emb[:1000]))

# 获取最后一个向量
matrix = tri_emb
last_vector = matrix[-1]

# 计算余弦相似度
dot_product = torch.matmul(matrix, last_vector)  # 点积
norm_matrix = torch.norm(matrix, dim=1)  # 每个向量的范数
norm_last = torch.norm(last_vector)  # 最后一个向量的范数

cosine_similarity = dot_product / (norm_matrix * norm_last)  # 计算余弦相似度
print(cosine_similarity)
import ipdb;ipdb.set_trace()