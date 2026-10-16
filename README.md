# 课程/专业文档 RAG 问答系统

这是一个基于 RAG（检索增强生成）的智能问答应用，支持上传 PDF / TXT 文档，实现基于本地知识库的精准问答，并附带引用溯源功能。

## ✨ 核心功能
- **文档上传与解析**：支持 PDF 和 TXT 格式，自动解析文本。
- **文本切分**：采用定长 + 重叠（Overlap）策略，保证语义完整性。
- **向量检索**：使用 ChromaDB 存储向量，基于 `BAAI/bge-small-zh` 模型实现语义检索。
- **生成与溯源**：调用 DeepSeek API 生成回答，并在回答下方展示引用来源，有效抑制幻觉。
- **界面**：基于 Streamlit 构建的聊天式网页界面。

## 🛠️ 技术栈
- Python 3.12
- Streamlit (Web 界面)
- ChromaDB (向量数据库)
- Sentence-Transformers (Embedding)
- OpenAI SDK (调用 DeepSeek API)
- PyPDF (PDF 解析)

## 🚀 如何运行
1. 克隆仓库到本地：`git clone https://github.com/rainin067/my-python-project.git`
2. 安装依赖包：`pip install -r requirements.txt`
3. 启动应用：`streamlit run app.py`
4. 在浏览器打开 `http://localhost:8501`，输入你的 DeepSeek API Key 即可开始提问。

## 📸 效果截图
![RAG问答系统界面](https://github.com/user-attachments/assets/cbe05c7c-3649-4ad4-825b-3e448aa87b01)
