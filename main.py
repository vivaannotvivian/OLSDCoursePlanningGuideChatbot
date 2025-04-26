import os
import json
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.llms import Ollama
import fitz  # PyMuPDF

def load_pdf(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.getvalue())
    loader = PyPDFLoader("temp.pdf")
    return loader.load()

def get_pdf_first_page_image(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.getvalue())
    doc = pymupdf.open("temp.pdf")
    os.makedirs("static", exist_ok=True)  # Ensure storage directory exists
    pix = doc[0].get_pixmap()
    image_path = "static/first_page.png"
    pix.save(image_path)
    return image_path

def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(docs)

def create_retriever(docs_json):
    docs = json.loads(docs_json)
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector = FAISS.from_texts(
        [doc["page_content"] for doc in docs],
        embedder,
        metadatas=[{"source": doc["metadata"].get("source", "Uploaded PDF")} for doc in docs]
    )
    return vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def build_prompt():
    prompt = """
    1. Use the following context to answer the question.
    2. If you don't know the answer, say "I don't know."
    3. Keep the answer concise (3-4 sentences).
    
    Context: {context}
    
    Question: {question}
    
    Helpful Answer:"""
    return PromptTemplate.from_template(prompt)

def build_qa_chain(retriever, llm):
    prompt = build_prompt()
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nsource:{source}",
    )
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
    )
    return RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        retriever=retriever,
        return_source_documents=True,
    )

def main():
    st.set_page_config(layout="wide")
    st.title("🚀 Fast RAG-based QA with DeepSeek R1")
    
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        
        if uploaded_file:
            try:
                image_path = get_pdf_first_page_image(uploaded_file)
                st.image(image_path, caption="First Page Preview", use_column_width=True)
            except Exception as e:
                st.error("Failed to load preview: " + str(e))
    
    if uploaded_file:
        with st.spinner("🔄 Processing PDF..."):
            retriever = process_pdf(uploaded_file)
        
        llm = get_llm()
        qa_chain = build_qa_chain(retriever, llm)
        
        user_input = st.text_input("Enter your question:")
        
        if user_input:
            with st.spinner("🤖 Generating response..."):
                response = qa_chain.invoke({"query": user_input})["result"]
                st.write("### 📜 Answer:")
                st.write(response)
    else:
        st.info("📥 Please upload a PDF file to proceed.")

