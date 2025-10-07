from langchain.document_loaders import PyPDFLoader, DirectoryLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

# Extract Data From PDF Directory
def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# Extract Data From CSV
def load_csv_file(file_path):
    loader = CSVLoader(file_path=file_path, encoding="utf-8")
    documents = loader.load()
    return documents

# Keep minimal metadata
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

# Split into text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Multilingual embeddings (English + Tamil)
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name='intfloat/multilingual-e5-large'  # 1024 dimensions
    )
    return embeddings
