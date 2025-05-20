# pip install streamlit openai faiss-cpu langchain langchain_community

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

# Cấu hình Streamlit
st.set_page_config(page_title="FPT Chatbot", layout="centered")
st.title("🤖 Chatbot hỗ trợ tuyển sinh FPT Polytechnic")

api_key = st.secrets["OPENAI_API_KEY"]

def load_retriever():
    with open("tuyensinh.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(texts, embedding=embeddings)

    return vectorstore.as_retriever()

retriever = load_retriever()

# Load mô hình GPT từ OpenAI
llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo", temperature=0.7)

# Tạo chuỗi RAG
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

# Hàm sinh câu trả lời
def generate_answer(query):
    result = qa_chain.run(query)
    return result.strip()


# Giao diện người dùng
user_input = st.text_input("Nhập câu hỏi của bạn:")

if st.button("Trả lời"):
    if user_input:
        with st.spinner("Đang truy xuất và sinh câu trả lời..."):
            answer = generate_answer(user_input)
            st.success("✅ Câu trả lời:")
            st.write(answer)
    else:
        st.warning("❗ Vui lòng nhập câu hỏi.")
