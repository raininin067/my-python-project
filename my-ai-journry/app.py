#app.py 的目的，是把你之前写的“黑漆漆的终端脚本”，变成一个有界面、能交互、可展示给面试官看的“完整产品”。
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pypdf import PdfReader

# 设置页面标题和图标
st.set_page_config(page_title="课程文档RAG问答", page_icon="📚")
st.title("📚 课程/专业文档 RAG 问答系统")

# 1. 侧边栏：安全输入 API Key（避免密钥泄露到代码里）
with st.sidebar:
    st.header("🔑 配置")
    api_key = st.text_input("请输入 DeepSeek API Key:", type="password")
    st.markdown("---")
    st.markdown("**使用说明：**\n1. 输入 API Key\n2. 上传 PDF 文档\n3. 开始提问")

# 2. 上传文档并初始化知识库
uploaded_file = st.file_uploader("上传你的 PDF 或 TXT 文件", type=["pdf", "txt"])


# 用 Streamlit 的缓存机制，避免每次提问都重新加载模型和文档
@st.cache_resource
def load_models():
    return SentenceTransformer('BAAI/bge-small-zh-v1.5')


if uploaded_file is not None:
    # 初始化模型和数据库
    embed_model = load_models()

    if "collection" not in st.session_state:
        client_db = chromadb.Client()
        st.session_state.collection = client_db.create_collection(name="knowledge")

        # 解析文件
        if uploaded_file.name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        else:
            text = uploaded_file.read().decode("utf-8")


        # 切分并入库
        def split_text(t, chunk_size=200, overlap=50):
            chunks = []
            start = 0
            while start < len(t):
                end = start + chunk_size
                chunks.append(t[start:end])
                start = end - overlap
            return chunks


        chunks = split_text(text)
        ids = [f"doc_{i}" for i in range(len(chunks))]
        st.session_state.collection.add(documents=chunks, ids=ids)
        st.success(f"✅ 文档已处理完毕，共切分 {len(chunks)} 个知识块！")

# 3. 问答界面
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户输入
if prompt := st.chat_input("请输入关于文档的问题..."):
    if not api_key:
        st.warning("请先在左侧输入 API Key！")
    else:
        # 显示用户问题
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 检索 + 生成
        with st.chat_message("assistant"):
            with st.spinner("正在查阅文档并思考..."):
                results = st.session_state.collection.query(query_texts=[prompt], n_results=2)
                retrieved_docs = results['documents'][0]
                context_text = "\n".join(retrieved_docs)

                # 拼接提示词
                final_prompt = f"请严格根据以下参考资料回答问题。如果资料里没有，直接回答不知道。\n\n【参考资料】：\n{context_text}\n\n【问题】：{prompt}"

                llm_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                response = llm_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": final_prompt}]
                )
                answer = response.choices[0].message.content

                # 输出回答和引用
                st.markdown(answer)
                with st.expander("📚 查看引用来源"):
                    for i, doc in enumerate(retrieved_docs):
                        st.write(f"{i + 1}. {doc}")

                st.session_state.messages.append({"role": "assistant", "content": answer})
#面试官问：“你项目 A 的前端是怎么做的？”你的回答：“我使用了 Streamlit 框架进行快速原型开发。我没有把 API Key 硬编码在代码里，而是通过网页侧边栏输入，
# 兼顾了安全性与用户体验。同时我利用了 Streamlit 的 session_state 管理对话历史，并使用了 @st.cache_resource 缓存向量模型，避免了重复加载，
# 保证了网页的响应速度。用户可以直接上传 PDF，系统会自动进行解析和向量入库。”