# chatbot_api.py
import torch
import re
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter as rcts
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from transformers import AutoTokenizer
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import retrieval_qa
from transformers import AutoModelForCausalLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
def get_last_line(text):
  lines = text.splitlines()
  if lines:
    return lines[-1]
  return "Answer not found."
app = Flask(__name__)
CORS(app, origins="null")
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    
    # Replace with your actual response logic
    response = generate_response(user_input)
    
    return jsonify({"reply": response})

def generate_response(textbox_input):
    print("Current working directory:", os.getcwd())
    print("Files in this directory:", os.listdir())
    # Simple example: replace with your real Python logic
    system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "End your answer with this format -> Final Answer: {{your answer here}}, and wrap ONLY that part in curly braces.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)
    prompt = ChatPromptTemplate.from_template(system_prompt)
    file_path = os.path.join(os.path.dirname(__file__), "guide.txt")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print("loaded text")

    text_splitter = rcts(
    chunk_size=500,
    chunk_overlap=50,
)

    docs = text_splitter.split_documents(documents)
    print("split text")
    embeddings = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(docs, embeddings)
    print("faissed the thing")

    tokenizer_b = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")
    model_b = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-chat", torch_dtype=torch.bfloat16, device_map="cuda", temperature = 0.6, do_sample = True)
    print("got model and tokenizer")

    hf_pipline_vivaan = pipeline("text-generation", model = model_b, tokenizer=tokenizer_b, max_new_tokens=500, do_sample=True)

    llm = HuggingFacePipeline(pipeline=hf_pipline_vivaan)
    print("found llm")
    retriever = db.as_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_stuff_documents_chain(llm, prompt)
    print("chain created")
    #relevant_docs = retriever.invoke(textbox_input)
    relevant_docs_with_scores = retriever.vectorstore.similarity_search_with_relevance_scores(textbox_input.strip())
    relevant_docs = [doc[0] for doc in relevant_docs_with_scores]
    print(relevant_docs)
    print("found context")
    query = textbox_input
    answer = chain.invoke({"context": relevant_docs, "question": query})
    #print("Answer: ", answer)
    return_var = get_last_line(answer)
    print(answer)
    return return_var

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
