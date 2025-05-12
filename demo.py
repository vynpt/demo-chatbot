# pip install streamlit faiss-cpu sentence-transformers langchain transformers langchain_community

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# --- Load data và tạo vector store ---
@st.cache_resource
def load_retriever():
    # Đọc nội dung file
    with open("tuyensinh.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Chia đoạn
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(raw_text)

    # Embedding
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    db = FAISS.from_texts(texts, embeddings)

    return db.as_retriever()

retriever = load_retriever()

# --- Load LLM ---
@st.cache_resource
def load_llm():
    generator = pipeline("text-generation", model="microsoft/phi-2", tokenizer="microsoft/phi-2")
    return generator

generator = load_llm()

# --- Hàm trả lời ---
def generate_answer(query):
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs[:3]])

    prompt = f"""Dựa trên ngữ cảnh sau, hãy trả lời câu hỏi:
                Ngữ cảnh:
                {context}
                Câu hỏi:
                {query}
                Trả lời: """

    result = generator(prompt, max_new_tokens=200, do_sample=True)[0]["generated_text"]
    return result[len(prompt):].strip()  # loại bỏ phần prompt gốc

# --- Giao diện Streamlit ---
st.set_page_config(page_title="Truong ...", layout="centered")
st.title("🤖 Chatbot ho tro tuyen sinh")

user_input = st.text_input("Nhập câu hỏi của bạn:")

if st.button("Trả lời"):
    if user_input:
        with st.spinner("Đang truy xuất và sinh câu trả lời..."):
            answer = generate_answer(user_input)
            st.success("✅ Câu trả lời:")
            st.write(answer)
    else:
        st.warning("❗ Vui lòng nhập câu hỏi.")

