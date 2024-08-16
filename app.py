from flask import Flask, request, render_template, jsonify
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA

app = Flask(__name__)

# Function to extract PDF data
def extract_pdf_data(directory):
    loader = DirectoryLoader(
        path=directory,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# Function to split text into chunks
def split_text_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# Function to fetch Hugging Face embeddings
def fetch_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    return embeddings

# Load and prepare data
data = extract_pdf_data("data/")
chunks = split_text_into_chunks(data)
embeddings = fetch_hugging_face_embeddings()
DB_FAISS_PATH = 'vectstore/db'
db = FAISS.from_documents(chunks, embeddings)
db.save_local(DB_FAISS_PATH)

# Prepare the LLM and RetrievalQA chain
prompt_template = """
You are provided with some context and a question. Your task is to use the context to answer the user's question accurately and concisely.

Follow these instructions:

1. Carefully read the context and identify the key information relevant to the question.
2. Summarize the relevant parts of the context in your mind before formulating your answer.
3. Ensure your answer is factually accurate based on the provided context.
4. If you are unsure about any part of the answer, it's better to say "I don't know I can provide only Medical information " than to provide an incorrect answer.
5. Keep your answer concise and to the point.

Context: {context}
Question: {question}

Helpful answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

model = CTransformers(model="/home/mg/lifeCare/model/llama-2-7b-chat.ggmlv3.q2_K.bin",
                      model_type="llama",
                      config={'max_new_tokens': 512, 'temperature': 0.8})

qa = RetrievalQA.from_chain_type(llm=model,
                                 chain_type='stuff',
                                 retriever=db.as_retriever(search_kwargs={'k': 2}),
                                 return_source_documents=True,
                                 chain_type_kwargs=chain_type_kwargs
                                 )

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['prompt']
    result = qa({"query": user_input})
    response = result["result"]
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
