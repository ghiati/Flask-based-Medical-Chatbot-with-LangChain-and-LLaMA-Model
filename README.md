# Flask-based Medical Chatbot with LangChain and LLaMA Model

This project is a Flask-based web application that leverages LangChain, FAISS, and a LLaMA model for creating a medical chatbot. The chatbot answers users' questions by retrieving relevant information from pre-loaded documents and providing factually accurate responses based on that context. The project uses sentence embeddings for document retrieval and LLMs for generating the answers.

## Features
- **Retrieval-Based QA**: Uses FAISS for document retrieval based on user queries.
- **LLM-Powered Responses**: Integrates a pre-trained LLaMA language model for generating concise and contextually accurate answers.
- **Flask Web Application**: Provides a simple interface for users to interact with the chatbot.
- **Customizable Prompt Template**: Tailored prompt design to ensure factually accurate and concise responses.

## Tech Stack
- **Flask**: For creating the web application.
- **LangChain**: For document loading, splitting, and managing the language model chain.
- **FAISS**: For efficient document vector search and retrieval.
- **HuggingFace Sentence Transformer**: For generating embeddings from documents.
- **LLaMA Model (CTransformers)**: For generating the answer to user questions.

## Project Structure
```
.
├── data
│   └── Medical_book_compressed.pdf  # PDF file containing medical information
├── lifebot
│   └── model
│       └── llama-2-7b-chat.ggmlv3.q2_K.bin  # Pre-trained LLaMA model
├── templates
│   └── index.html        # HTML file for the web interface
├── vectstore
│   └── db
│       ├── index.faiss   # FAISS index file
│       └── index.pkl     # FAISS metadata
├── .gitignore            # Git ignore file
├── app.py                # Main Flask application
├── data_preparing.py      # Python script for data preparation
├── data_preprocessing.ipynb # Jupyter notebook for data preprocessing
├── README.md             # Project documentation (this file)
├── requirements.txt      # Python dependencies

```

### Install Dependencies
To install the required dependencies, run:
```bash
pip install requirement.txt
```

### Running the Application
1. Clone the repository and navigate to the project directory.
2. Place your FAISS vector store and LLaMA model in the specified directories.
3. Run the Flask application:
   ```bash
   python app.py
   ```
4. Open your browser and navigate to `http://127.0.0.1:5000/` to use the chatbot.

### How It Works
- **Document Loader**: Loads PDF documents into the system using `PyPDFLoader`.
- **Embeddings**: Generates sentence embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- **FAISS Retrieval**: Searches for relevant documents based on user queries.
- **LLM (LLaMA)**: Answers user queries based on the retrieved context using the LLaMA model.

### Usage
1. Start the app.
2. Enter a medical query in the provided input field.
3. The chatbot retrieves relevant information from the document store and generates a concise answer.

### Customization
- **Model**: Replace the `LLaMA` model with another model by changing the `CTransformers` configuration.
- **Prompt Template**: Customize the prompt in `PromptTemplate` to adjust the answering style or behavior.
- **Document Store**: Add new documents to the FAISS vector store by using the appropriate document loaders and splitters.

## License
This project is licensed under the MIT License.

