import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("正在加载模型，第一次运行会下载几百MB，请耐心等待...")
# 加载轻量级中文向量模型
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
print("模型加载完成！")

# 准备三句话
sentences = [
    "我今天买了一斤苹果，很甜",
    "刚从水果店买了个红富士，味道不错",
    "这辆汽车的发动机坏了"
]

# 1. 把三句话变成向量（一堆数字）
embeddings = model.encode(sentences)
print(f"\n第一句话的向量前5个数字是：{embeddings[0][:5]}")

# 2. 计算相似度
sim_0_1 = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
sim_0_2 = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]

print(f"\n【苹果 vs 水果店红富士】的相似度：{sim_0_1:.4f} （越接近1越像）")
print(f"【苹果 vs 汽车发动机】的相似度：{sim_0_2:.4f} （越接近0越无关）")