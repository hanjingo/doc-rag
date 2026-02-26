from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Load documents from the "data" directory
DATA_PATH = "./3rd/doc"

# Create a DirectoryLoader to load all text files from the specified directory
loader = DirectoryLoader(DATA_PATH, silent_errors=True)
raw_documents = loader.load()
print(f"loaded {len(raw_documents)} documents from {DATA_PATH}")

# Create a RecursiveCharacterTextSplitter to split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Maximum size of each chunk
    chunk_overlap=200, # Number of characters to overlap between chunks
    separators=["\n\n", "\n", " ", ""] # List of separators to use for splitting the text
)

# Split the raw documents into smaller chunks
documents = text_splitter.split_documents(raw_documents)
print(f"split into {len(documents)} chunks")

# Create a HuggingFaceEmbeddings instance using the specified embedding model
embedding_model_name = 'BAAI/bge-small-zh-v1.5'

# Initialize the HuggingFaceEmbeddings with the specified model
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"device": "cpu"},  # Use CPU for embedding generation
    encode_kwargs={"normalize_embeddings": True} # Normalize the embeddings to have unit length
)
print(f"initialized HuggingFaceEmbeddings with model: {embedding_model_name}")

# Create a FAISS vector store from the documents and their corresponding embeddings
vectorstore = FAISS.from_documents(documents, embeddings)

# Save the FAISS index to a local directory
#vectorstore.save_local("faiss_index_directory") 
#vectorstore = FAISS.load_local("faiss_index_directory", embeddings, allow_dangerous_deserialization=True)

# Create a retriever from the FAISS vector store to enable retrieval of relevant documents based on queries
retriever = vectorstore.as_retriever()
print("FAISS vector store created and retriever initialized")

# Initialize the Ollama language model
llm = Ollama(model="gpt-3.5-turbo")

# Create a RetrievalQA chain using the retriever and the Ollama language model
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # The "stuff" chain type concatenates all retrieved documents into a single input for the language model
    retriever=retriever # The retriever is used to fetch relevant documents based on the user's query
)

# Example query to test the RetrievalQA chain
query = "What is the main topic of the documents?"
answer = qa_chain.run(query)
print(f"Query: {query}\nAnswer: {answer}")
