# pip install streamlit faiss-cpu sentence-transformers langchain transformers langchain_community

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    tokenizer = AutoTokenizer.from_pretrained("VietAI/gpt-neo-1.3B-vietnamese-news")
    model = AutoModelForCausalLM.from_pretrained("VietAI/gpt-neo-1.3B-vietnamese-news")
    return tokenizer, model

tokenizer, model = load_model()

# Hàm sinh câu trả lời 
def generate_answer(query):
    docs = retriever.get_relevant_documents(query)

    if not docs:
        return "Xin lỗi, tôi không tìm thấy thông tin phù hợp để trả lời câu hỏi này."

    context = "\n".join([doc.page_content for doc in docs[:3]])
    
    if not docs or len(docs) == 0:
        return "Xin lỗi, tôi không tìm thấy thông tin phù hợp để trả lời câu hỏi này."

    prompt = f"""Dựa trên ngữ cảnh sau, hãy trả lời câu hỏi một cách ngắn gọn và dễ hiểu:
            Ngữ cảnh:
            {context}

            Câu hỏi:
            {query}

            Trả lời:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            temperature=0.8,
            repetition_penalty=1.2,  # giảm lặp lại
            eos_token_id=tokenizer.eos_token_id  # dừng tại kết thúc câu
            )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = response[len(prompt):].strip()
    cleaned = generated.split("\n")[0].strip()
    return cleaned


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
