import torch
import re
import streamlit as st
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

def delete_between_phrases(text, start_phrase, end_phrase):
    pattern = re.escape(start_phrase) + r".*?" + re.escape(end_phrase)
    modified_text = re.sub(pattern, "", text, flags=re.DOTALL)
    return modified_text

def ai_process(textbox_input):
    system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use ten sentence maximum and keep the answer concise. "
    "The answer for the user to read should begin with 'Beginning:' and end with 'End' "
    "Context: {context}"
)
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

    loader = TextLoader("guide.txt", encoding='utf-8')
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
    model_b = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-chat", torch_dtype=torch.bfloat16, device_map="auto")
    print("got model and tokenizer")

    hf_pipline_vivaan = pipeline("text-generation", model = model_b, tokenizer=tokenizer_b, max_new_tokens=500, do_sample=True)

    llm = HuggingFacePipeline(pipeline=hf_pipline_vivaan)
    print("found llm")
    retriever = db.as_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    print("chain created")
    query = textbox_input
    answer = chain.invoke({"input": query})
    print("Answer: ", answer["answer"])
    return_var = answer["answer"]
    return return_var
def main():
    st.set_page_config(layout="wide")
    st.title("OLSD Course Planning Chatbot🤖")

    user_inputbox = st.text_input("❔ Enter your question:")
    if user_inputbox:
            with st.spinner("✎ Generating response..."):
                response = ai_process(user_inputbox)
                mod_response = delete_between_phrases(response, "System:", "Beginning:")
                st.write("### 📜 Answer:")
                st.write(response)
                st.write('If no answer is present, please reword your question.')
if __name__ == "__main__":
    main()

    
