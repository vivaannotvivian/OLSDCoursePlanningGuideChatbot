from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter as rcts
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from transformers import AutoTokenizer
from langchain_community.llms import huggingface_pipeline
from langchain.chains import retrieval_qa
from transformers import AutoModelForCausalLM


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

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
print("got model and tokenizer")

hf_pipline = pipeline("text2text-generation", model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", max_new_tokens=500, temperature=0.7, do_sample=True)

llm = huggingface_pipeline(pipeline=hf_pipline)
print("found llm")
qa_chain = retrieval_qa.from_chain_type(
    llm=llm,
    retriever=db.as_retriever()
)
print("made qa chain")
query = "What are all the math classes offered?"

answer = qa_chain.run(query)
print("Answer: ", answer)

