from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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
