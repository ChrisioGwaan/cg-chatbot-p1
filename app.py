from fastapi import FastAPI, APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
from sqlalchemy import create_engine, and_, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import uvicorn
import os
import uuid
import time
from chat_completion import completion, detect_intention
from models import Base, Message, Product, ParentDocument, DocumentChunk
from openai import AzureOpenAI
from azure.cosmos import CosmosClient
from urllib.parse import quote
from models.all_models import Conversation

app = FastAPI(
    title="FastAPI Service",
    description="FastAPI service with SQLAlchemy ORM",
    version="0.1.0",
)
router = APIRouter()


def make_response(code, msg, result=None) -> JSONResponse:
    return JSONResponse(
        status_code=code, content={"code": code, "msg": msg, "result": result}
    )


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

PARTITION_KEY_PATH = "/id"
VECTOR_FIELD = "embedding"
CONTENT_FIELD = "chunk_text"
METADATA_FIELD = "group_category"

openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION,
)
cosmos_client = CosmosClient.from_connection_string(COSMOSDB_CONNECTION_STRING)
database = cosmos_client.get_database_client(COSMOSDB_DATABASE_NAME)
container = database.get_container_client(COSMOSDB_CONTAINER_NAME)

DATABASE_URL = f"sqlite:///{db_path}" # TODO: migrate to PostgreSQL
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

Base.metadata.create_all(bind=engine)


# Pydantic Models
class ProductModel(BaseModel):
    product_id: str
    scheme_code: str
    product_description: str
    is_active: bool

    class Config:
        orm_mode = True


class ParentDocumentModel(BaseModel):
    id: str
    owner_id: str
    file_name: str
    classification: str
    fk_product_id: str

    class Config:
        orm_mode = True


class ChunkModel(BaseModel):
    document_chunk_id: str
    parent_document_id: str
    page_index: int
    content: str
    token_count: int

    class Config:
        orm_mode = True


class TextChunkUploadRequest(BaseModel):
    products: Optional[ProductModel] = None
    parent_document: ParentDocumentModel
    chunks: List[ChunkModel]


class DocumentStatusUpdate(BaseModel):
    status: str


class IntentionDetectionRequest(BaseModel):
    user_query: str
    products: str


# Request Model
class CompletionRequest(BaseModel):
    user_id: str
    conversation_id: str
    content: str
    product_name: str
    role: Optional[str] = "user"
    top_k: Optional[int] = 5  # default value for top_k

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_001",
                "conversation_id": "conv_123",
                "content": "What is AI?",
                "product_name": "E1-BizJamin",
                "role": "user",
                "top_k": 5,
            }
        }


# Response Model
class CompletionResponse(BaseModel):
    id: str
    message_id: str
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    created_at: datetime

    class Config:
        orm_mode = True


class UserEmailRequest(BaseModel):
    email: str


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_next_sequence(db: Session, conversation_id: str) -> int:
    """Get the sequence number of the next message in the session"""
    last_msg = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.sequence.desc())
        .first()
    )
    return (last_msg.sequence + 1) if last_msg else 1


def get_query_embedding(query: str) -> list:
    """Generate embedding for the user query using Azure OpenAI."""
    response = openai_client.embeddings.create(
        input=query, model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
    )
    return response.data[0].embedding


def safe_url(url: str) -> str:
    if "?" in url:
        base, query = url.split("?", 1)
        return quote(base, safe="/:") + "?" + query
    return quote(url, safe="/:")


@router.get("/")
async def read_root():
    return {"message": "Yes to SQLAlchemy!"}


# @router.post("/upload-text-chunks/")
# async def upload_text_chunks(
#     request: TextChunkUploadRequest, db: Session = Depends(get_db)
# ):
#     try:
#         # Process product if exists
#         if request.products:
#             db_product = Product(
#                 product_id=request.products.product_id,
#                 scheme_code=request.products.scheme_code,
#                 product_description=request.products.product_description,
#                 is_active=request.products.is_active,
#             )
#             db.add(db_product)
#             db.flush()  # Ensure product_id is available for foreign key

#         # Create parent document
#         db_document = ParentDocument(
#             id=request.parent_document.id,
#             owner_id=request.parent_document.owner_id,
#             file_name=request.parent_document.file_name,
#             classification=request.parent_document.classification,
#             fk_product_id=request.parent_document.fk_product_id,
#         )
#         db.add(db_document)

#         # Create chunks
#         for chunk in request.chunks:
#             db_chunk = DocumentChunk(
#                 document_chunk_id=chunk.document_chunk_id,
#                 parent_document_id=chunk.parent_document_id,
#                 page_index=chunk.page_index,
#                 content=chunk.content,
#                 token_count=chunk.token_count,
#             )
#             db.add(db_chunk)

#         db.commit()
#         return {
#             "status": "success",
#             "document_id": request.parent_document.id,
#             "chunk_count": len(request.chunks),
#             "product_processed": request.products is not None,
#         }

#     except Exception as e:
#         db.rollback()
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Database error: {str(e)}",
#         )


# @router.put("/update-document-status/{document_id}")
# async def update_document_status(
#     document_id: str, request: DocumentStatusUpdate, db: Session = Depends(get_db)
# ):
#     db_document = (
#         db.query(ParentDocument).filter(ParentDocument.id == document_id).first()
#     )
#     if not db_document:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
#         )

#     try:
#         db_document.ingest_status = request.status
#         db.commit()
#         return {
#             "status": "success",
#             "document_id": document_id,
#             "new_status": request.status,
#         }
#     except Exception as e:
#         db.rollback()
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
#         )


@router.post("/chat-stream")
async def chat_stream_endpoint(
    request: CompletionRequest, db: Session = Depends(get_db)
):
    """Detect user intent to classify the query into one of the predefined categories."""

    start_time = time.perf_counter()

    user_id = request.user_id
    conversation_id = request.conversation_id
    user_query = request.content
    product_name = request.product_name
    top_k = request.top_k or 5

    if not user_id or not conversation_id:
        return make_response(
            400, "Missing required fields: user_id and conversation_id", None
        )

    if not user_query:
        return make_response(400, "Missing required fields: query", None)

    if not top_k:
        return make_response(400, "Missing required field: top_k", None)

    message = Message(
        id=str(uuid.uuid4()),
        user_id=request.user_id,
        conversation_id=request.conversation_id,
        content=request.content,
        role=request.role,
        sequence=get_next_sequence(db, request.conversation_id),
        created_at=datetime.now(),
    )

    product_list = []
    if product_name:
        scheme_code, product_description = product_name.split("-", 1)

        product = (
            db.query(Product)
            .filter(
                and_(
                    Product.scheme_code == scheme_code,
                    Product.product_description == product_description,
                )
            )
            .first()
        )

        product_id = product.product_id if product else None
        product_list = [product_id] if product_id else []

        # Exception if product_name is invalid
        if len(product_list) == 0:
            return make_response(400, "Invalid product_name", None)
    else:
        # query = text(
        #     """
        #     SELECT DISTINCT
        #         a.product_id
        #     FROM
        #         users d
        #     INNER JOIN user_orgs e ON d.id = e.user_id
        #     INNER JOIN user_roles c ON d.id = c.user_id
        #     INNER JOIN product_role_permissions b ON c.role_id = b.role_id
        #     INNER JOIN products a ON b.product_id = a.product_id
        #     INNER JOIN product_orgs f ON a.product_id = f.fk_product_id AND e.org_id = f.fk_org_id
        #     WHERE
        #         d.id = :user_id
        #     """
        # )

        with SessionLocal() as db:
            result = db.execute(query, {"user_id": user_id}).fetchall()

            product_list = [(row.product_id) for row in result]

        # Exception if no products found for the user
        if len(product_list) == 0:
            return make_response(400, "No products found for the user", None)

    completion_result = completion(db, product_list, message, request.top_k)

    if isinstance(completion_result, str):
        return make_response(400, completion_result, None)

    message.final_completion_id = completion_result.id

    db.add(message)
    db.add(completion_result)
    db.commit()

    end_time = time.perf_counter()

    print(f"\n\nTotal tokens used")
    print(f"Time taken: {end_time - start_time:.2f} seconds\n\n")

    return make_response(200, "Query successfully processed", completion_result.answer)


@router.post("/initiate-conversation")
async def initiate_conversation(
    request: UserEmailRequest, db: Session = Depends(get_db)
):
    query = text(
        """
        SELECT
            *
        FROM
            users a
        WHERE
            a.email = :email
    """
    )

    try:
        result = db.execute(query, {"email": request.email}).fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="User not found")

        new_conversation = Conversation(
            id=str(uuid.uuid4()),
            user_id=result.id,
            created_at=datetime.now(),
        )

        db.add(new_conversation)
        db.commit()
        db.refresh(new_conversation)

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Error initiating conversation")

    return {"conversation_id": new_conversation.id, "user_id": new_conversation.user_id}


@router.post("/retrieve-products")
async def retrieve_products_by_user_email(
    request: UserEmailRequest, db: Session = Depends(get_db)
) -> str:
    try:
        # query = text(
        #     """
        # SELECT DISTINCT a.product_id, a.scheme_code, a.product_description
        # FROM users d
        # INNER JOIN user_orgs e ON d.id = e.user_id
        # INNER JOIN user_roles c ON d.id = c.user_id
        # INNER JOIN product_role_permissions b ON c.role_id = b.role_id
        # INNER JOIN products a ON b.product_id = a.product_id
        # INNER JOIN product_orgs f ON
        #     a.product_id = f.fk_product_id
        #     AND e.org_id = f.fk_org_id
        # WHERE d.email = :email
        # """
        # )

        rows = db.execute(query, {"email": request.email}).fetchall()

        if not rows:
            return "Sorry, you do not have access to any products."

        return ", ".join(
            f"{(row.scheme_code or '').strip()}-{(row.product_description or '').strip()}"
            for row in rows
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving products: {str(e)}",
        )


@router.post("/intention-detection")
async def intention_detection(
    request: IntentionDetectionRequest, db: Session = Depends(get_db)
):
    try:
        response = detect_intention(request.user_query, request.products, db=db)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"intent error: {e}",
        )

    return response


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
