# pip install streamlit faiss-cpu sentence-transformers langchain transformers langchain_community

import streamlit as st
from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Cấu hình Streamlit
st.set_page_config(page_title="FPT Chatbot", layout="centered")
st.title("🤖 Chatbot hỗ trợ tuyển sinh FPT Polytechnic")

# Load dữ liệu và tạo vector store
@st.cache_resource
def load_retriever():
    with open("tuyensinh.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    db = FAISS.from_texts(texts, embedding=embeddings)

    return db.as_retriever()

retriever = load_retriever()

# Load mô hình sinh tiếng Việt 
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="NlpHUST/gpt2-vietnamese", repetition_penalty=1.3)

generator = load_model()

# Hàm sinh câu trả lời
def generate_answer(query):
    docs = retriever.get_relevant_documents(query)

    if not docs:
        return "Xin lỗi, tôi không tìm thấy thông tin phù hợp để trả lời câu hỏi này."

    context = "\n".join([doc.page_content for doc in docs[:3]])

    prompt = f"""Bạn là trợ lý tuyển sinh FPT Polytechnic. Chỉ sử dụng thông tin trong phần NGỮ CẢNH để trả lời câu hỏi."
                NGỮ CẢNH: {context}
                CÂU HỎI: {query}
                TRẢ LỜI:
              """

    output = generator(prompt, max_new_tokens=100, do_sample=True, top_p=0.85, temperature=0.7, truncation=True)[0]["generated_text"]
    answer = output[len(prompt):].strip()

    for end in [".", "\n", "•", ":"]:
        if end in answer:
            answer = answer.split(end)[0] + end
            break

    return answer.strip()


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
