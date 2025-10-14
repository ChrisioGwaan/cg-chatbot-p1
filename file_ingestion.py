import os
import csv
import base64
import uuid
import time
from datetime import datetime
from io import BytesIO
from sqlalchemy import create_engine, and_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient
from openai import AsyncAzureOpenAI
from typing import List, Dict
import numpy as np
import tiktoken
from pathlib import Path
from pdf2image import convert_from_path
from models import Base, Product, ParentDocument, DocumentChunk
from azure_blob_storage import FILE_NAME, SAS_LINK, CSV_OUTPUT, save_sas_link
import asyncio

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
)
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
COSMOSDB_CONNECTION_STRING = os.getenv("COSMOSDB_CONNECTION_STRING")
COSMOSDB_DATABASE_NAME = os.getenv("COSMOSDB_DATABASE_NAME")
COSMOSDB_CONTAINER_NAME = os.getenv("COSMOSDB_CONTAINER_NAME")

# Validate environment variables
if not all(
    [
        AZURE_CONNECTION_STRING,
        CONTAINER_NAME,
        AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_API_KEY,
        OPENAI_API_VERSION,
        AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
        COSMOSDB_CONNECTION_STRING,
    ]
):
    raise ValueError(
        "One or more environment variables are missing. Check your .env file."
    )

# Initialize clients
async_openai_client = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=OPENAI_API_VERSION,
)
cosmos_client = CosmosClient.from_connection_string(COSMOSDB_CONNECTION_STRING)
database = cosmos_client.get_database_client(COSMOSDB_DATABASE_NAME)
container = database.get_container_client(COSMOSDB_CONTAINER_NAME)
encoding = tiktoken.encoding_for_model(AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME)

# SQLAlchemy Setup

DATABASE_URL = f"sqlite:///{db_path}" # TODO: migrate to PostgreSQL
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Create tables
Base.metadata.create_all(bind=engine)


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


DB = next(get_db())


def get_blob_client():
    """Initialize and return Azure Blob Service Client."""
    return BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)


def list_pdfs_in_container() -> List[Dict[str, str]]:
    """List all PDF files in the Azure Blob Storage container with their group_category metadata."""
    blob_service_client = get_blob_client()
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    pdf_files = []
    for blob in container_client.list_blobs(include=["metadata"]):
        if blob.name.endswith(".pdf"):
            group_category = (
                blob.metadata.get("group_category", "unknown")
                if blob.metadata
                else "unknown"
            )
            pdf_files.append({"name": blob.name, "group_category": group_category})
    print(f"Found {len(pdf_files)} PDFs: {[pdf['name'] for pdf in pdf_files]}")
    return pdf_files


def download_pdf(blob_name: str, download_path: str) -> None:
    """Download a PDF file from Azure Blob Storage."""
    blob_service_client = get_blob_client()
    blob_client = blob_service_client.get_blob_client(
        container=CONTAINER_NAME, blob=blob_name
    )
    with open(download_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())


async def extract_text_from_pdf(original_pdf_path: str) -> List[DocumentChunk]:
    """Extract pages from a PDF file and feed into LLM for image inference"""

    print("Enter extract_text_from_pdf method")

    pdf_path = Path(original_pdf_path).resolve()

    try:
        with open(pdf_path, "rb"):
            document_chunks = []
            # convert pdf to images
            images = convert_from_path(
                pdf_path, poppler_path=r"E:\poppler-25.07.0\Library\bin"
            )
            image_batch_triple = []
            for index, image in enumerate(images, start=1):
                # Encode current image
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                image_batch_triple.append(
                    (
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                        index,
                        pdf_path,
                    )
                )
            document_chunks = await process_images(image_batch_triple)
            return document_chunks
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""


async def process_images(image_batch_triple):
    tasks = [
        summarize_image(img, index, pdf_path)
        for (img, index, pdf_path) in image_batch_triple
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def summarize_image(img, index, pdf_path):
    start_time = time.perf_counter()
    prompt = """
    Please help me extract the contents of the image, including all text, structural information, elements and other information,
    I want to use it to build a RAG knowledge base.
    You should only return the contents, without any other words."""

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": [img]},
    ]

    response = await async_openai_client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME, messages=messages
    )

    token_count = response.usage.total_tokens
    summary = response.choices[0].message.content

    file_name = os.path.basename(pdf_path)

    parent_document = (
        DB.query(ParentDocument).filter(ParentDocument.file_name == file_name).first()
    )

    parent_document_id = parent_document.id if parent_document else None

    document_chunk = DocumentChunk(
        document_chunk_id=str(uuid.uuid4()),
        parent_document_id=parent_document_id,
        page_index=index,
        content=summary,
        token_count=token_count,
    )

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Time spent on image {index}: {elapsed_time:.4f} seconds")

    return document_chunk


def chunk_text(text: str, max_chunk_size: int = 500) -> List[str]:
    """Split text into smaller chunks for embedding and clean problematic characters."""
    text = "".join(c for c in text if c.isprintable()).strip()
    text = " ".join(text.split())

    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_length += len(word) + 1
        if current_length > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"Created {len(chunks)} chunks")
    return chunks


async def create_embeddings(chunks: List[str]) -> List[np.ndarray]:
    """Create embeddings for text chunks using Azure OpenAI's embedding model."""

    print("Enter create_embeddings method")

    try:
        embeddings = []
        for chunk in chunks:
            response = await async_openai_client.embeddings.create(
                input=chunk, model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
            )
            embedding = np.array(response.data[0].embedding)
            embeddings.append(embedding)
        return embeddings
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return []


def check_existing_chunk(
    pdf_name: str, chunk_index: int, product_id: str = None
) -> bool:
    """Check if a chunk already exists in Cosmos DB."""
    try:
        query = "SELECT * FROM c WHERE c.pdf_name = @pdf_name AND c.chunk_index = @chunk_index AND c.product_id = @product_id"
        params = [
            {"name": "@pdf_name", "value": pdf_name},
            {"name": "@chunk_index", "value": chunk_index},
            {"name": "@product_id", "value": product_id},
        ]

        results = container.query_items(
            query=query, parameters=params, enable_cross_partition_query=True
        )
        return len(list(results)) > 0
    except Exception as e:
        print(f"Error checking existing chunk for {pdf_name}, chunk {chunk_index}: {e}")
        return False


def store_embedding(
    pdf_name: str,
    chunk: DocumentChunk,
    embedding: np.ndarray,
    product_id: str,
) -> None:
    """Store embedding and metadata in Azure Cosmos DB if not already present."""
    try:
        if check_existing_chunk(pdf_name, chunk.page_index, product_id):
            print(
                f"Skipping duplicate chunk for {pdf_name}, chunk {chunk.page_index}, product_id: {product_id}"
            )
            return

        cosmos_chunk = {
            "id": chunk.document_chunk_id,
            "pdf_name": pdf_name,
            "chunk_index": chunk.page_index,
            "product_id": product_id,
            "embedding": embedding.tolist(),  # Convert NumPy array to list for JSON serialization
        }
        container.create_item(body=cosmos_chunk)
        print(
            f"Stored embedding for {pdf_name}, chunk {chunk.page_index}, product_id: {product_id}"
        )

    except Exception as e:
        print(f"Error storing embedding for {pdf_name}, chunk {chunk.page_index}: {e}")


def add_products(parent_folder_path: str):
    """Add products to the database."""

    for i, folder_name in enumerate(os.listdir(parent_folder_path), start=1):
        folder_full_path = os.path.join(parent_folder_path, folder_name)
        print(f"Processing folder: {folder_full_path}")
        if not os.path.isdir(folder_full_path):
            continue  # Skip non-folder items

        scheme_code = folder_name[:2]
        product_description = folder_name[3:]
        product_id = "product_" + str(i)

        existing = DB.query(Product).filter(Product.product_id == product_id).first()
        if existing:
            print(f"Product {product_id} already exists, skipping.")
            continue

        product = Product(
            product_id=product_id,
            scheme_code=scheme_code,
            product_description=product_description,
            is_active=True,
        )
        DB.add(product)

    DB.commit()


def add_parent_documents(csv_file: str):
    """Add parent documents to the database from a CSV file."""

    parent_docs = []

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row[FILE_NAME]
            sas_link = row[SAS_LINK]

            # parse filename
            if "/" in filename:
                folder, file_name = filename.split("/", 1)
            else:
                folder, file_name = "", filename

            print(f"Processing file: {file_name} in folder: {folder}")

            if "-" in folder:
                scheme_code, product_description = folder.split("-", 1)
            else:
                scheme_code, product_description = folder[:2], folder[2:]

            print(
                f"Scheme Code: {scheme_code}, Product Description: {product_description}"
            )

            product = (
                DB.query(Product)
                .filter(
                    and_(
                        Product.scheme_code == scheme_code,
                        Product.product_description == product_description,
                    )
                )
                .first()
            )

            fk_product_id = product.product_id if product else None

            doc = ParentDocument(
                id=str(uuid.uuid4()),
                owner_id="user_001",
                file_name=file_name,
                blob_link=sas_link,
                classification=scheme_code,
                ingest_status="pending",
                uploaded_at=datetime.now(),
                fk_product_id=fk_product_id,
            )

            # Check if ParentDocument already exists, in this phase we assume the product documents won't be updated
            existing = (
                DB.query(ParentDocument)
                .filter(
                    and_(
                        ParentDocument.classification == doc.classification,
                        ParentDocument.file_name == doc.file_name,
                    )
                )
                .first()
            )

            if existing:
                print(
                    f"ParentDocument named {doc.file_name} of product {doc.classification} already exists, skipping."
                )
                continue

            DB.add(doc)

            parent_docs.append(doc)

    DB.commit()

    # update ingest_status 为 finished
    for doc in parent_docs:
        doc.ingest_status = "finished"
    DB.commit()


async def main():
    parent_folder_path = r".\data\Bible_pdfs"

    add_products(parent_folder_path)

    save_sas_link(parent_folder_path, CSV_OUTPUT)

    add_parent_documents(CSV_OUTPUT)

    for folder_name in os.listdir(parent_folder_path):
        scheme_code = folder_name[:2]
        folder_full_path = os.path.join(parent_folder_path, folder_name)

        for file_name in os.listdir(folder_full_path):
            file_path = os.path.join(folder_full_path, file_name)
            print(f"Processing file: {file_path}")

            # Extract text
            document_chunks = await extract_text_from_pdf(file_path)
            print(f"scheme_code:{scheme_code}")
            print(len(document_chunks))
            print(f"document chunks of PDFs: \n{document_chunks}")

            # Chunk text
            # chunks = chunk_text(text, max_chunk_size=500)
            chunks: List[DocumentChunk] = document_chunks

            # Create and store embeddings
            try:
                chunk_contents: List[str] = [chunk.content or "" for chunk in chunks]

                embeddings = await create_embeddings(chunk_contents)

                if not embeddings or len(embeddings) != len(chunks):
                    print(f"Skipping {file_name} due to embedding issues.")
                    continue

                for chunk, embedding in zip(chunks, embeddings):
                    # Check if DocumentChunk already exists, in this phase we assume the product documents won't be updated
                    existing = (
                        DB.query(DocumentChunk)
                        .filter(
                            and_(
                                DocumentChunk.parent_document_id
                                == chunk.parent_document_id,
                                DocumentChunk.page_index == chunk.page_index,
                            )
                        )
                        .first()
                    )

                    parent_doc = (
                        DB.query(ParentDocument)
                        .filter(ParentDocument.id == chunk.parent_document_id)
                        .first()
                    )

                    if existing:
                        print(
                            f"DocumentChunk of Page {chunk.page_index} in document {parent_doc.file_name} already exists, skipping."
                        )
                        continue

                    print(
                        f"Chunk {chunk.page_index} embedding shape: {embedding.shape}"
                    )
                    print(f"Chunk {chunk.page_index} embedding sample: {embedding[:5]}")

                    parent_document = (
                        DB.query(ParentDocument)
                        .filter(ParentDocument.id == chunk.parent_document_id)
                        .first()
                    )

                    assert (
                        parent_document.classification == scheme_code
                    ), f"Scheme code mismatch: {parent_document.classification} != {scheme_code}"
                    assert (
                        parent_document.file_name == file_name
                    ), f"File name mismatch: {parent_document.file_name} != {file_name}"

                    # insert embeddings to CosmosDB
                    store_embedding(
                        file_name, chunk, embedding, parent_document.fk_product_id
                    )

                    # insert text chunk into SQLite
                    DB.add(chunk)

                DB.commit()
                print(
                    f"Processed {len(chunks)} chunks for {file_name} with embeddings."
                )
            except Exception as e:
                print(f"Failed to process embeddings for {file_name}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
