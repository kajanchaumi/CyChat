from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, load_csv_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load English PDFs
eng_pdfs = load_pdf_file(data='D:\CyChat\CyChat\data\d-en')

# Load Tamil PDFs
tamil_pdfs = load_pdf_file(data='D:\CyChat\CyChat\data\d-ta')

# Load CSV (Tamil/English)
csv_docs = load_csv_file(file_path='D:\CyChat\CyChat\data\d-ta\csv\data-ta.csv')

# Merge all docs
all_docs = eng_pdfs + tamil_pdfs + csv_docs
filter_data = filter_to_minimal_docs(all_docs)
text_chunks = text_split(filter_data)

# Embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "cychat"

# Create Pinecone index if missing
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1024,   # multilingual-e5-large
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# Upload documents
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
    upsert_kwargs={"batch_size": 50}  # smaller batch
)
