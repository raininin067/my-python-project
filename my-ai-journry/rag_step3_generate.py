import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 1. 初始化工具
print("正在加载模型...")
embed_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
print("模型加载完成！")

# 初始化 ChromaDB 和 大模型 API
client_db = chromadb.Client()
collection = client_db.create_collection(name="my_knowledge")

llm_client = OpenAI(
    api_key="sk-7f09c29542cd437996ddfcf65f944761",
    base_url="https://api.deepseek.com"
)

# 2. 建库并入库（复用之前的5句话）
documents = [
    "Python 是一种解释型、面向对象的高级编程语言。",
    "RAG 是检索增强生成，它先检索相关知识再生成回答。",
    "大模型微调是指在预训练模型基础上用特定数据继续训练。",
    "FastAPI 是一个现代、快速的 Python Web 框架。",
    "苹果是一种富含维生素的水果，常见的有红富士。"
]
ids = [f"doc_{i}" for i in range(len(documents))]
collection.add(documents=documents, ids=ids)

# 3. 用户提问
query = input("\n请输入你的问题：")

# 4. 【R】检索：去数据库找相关的 2 条资料
results = collection.query(query_texts=[query], n_results=2)
retrieved_docs = results['documents'][0] # 拿到找出来的资料列表

# 5. 【G】生成：拼接 Prompt 并调用大模型
# 把找出来的两句话用换行符拼成一段话
context_text = "\n".join(retrieved_docs)

# 这是最关键的一步：构建包含上下文的 Prompt！
prompt = f"""请严格根据以下参考资料回答用户的问题。
如果参考资料中没有相关信息，请直接回答“我不知道”，不要自己编造。

【参考资料】：
{context_text}

【用户问题】：
{query}
"""

print("\n正在请求大模型生成回答...")
response = llm_client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": prompt}]
)

# 6. 打印最终结果和引用来源
print("\n--- 🤖 AI 回答 ---")
print(response.choices[0].message.content)

print("\n--- 📚 引用来源 ---")
for i, doc in enumerate(retrieved_docs):
    print(f"{i+1}. {doc}")