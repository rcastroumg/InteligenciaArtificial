import json
import uuid
from fastapi import APIRouter, UploadFile, HTTPException, status
from schemas.ChatbotSchema import NewConversationInput, QueryInput
from utils.ChatbotUtil import RAGChatbotAPI
from typing import List
from models.mysql_models.conversation import ConversationModel
from models.mysql_models.conversation_history import ConversationHistoryModel
import PyPDF2
import io

router = APIRouter(
    prefix='/chatbot',
    tags=['Chatbot IA'],
)

rag_chatbot = RAGChatbotAPI()

@router.post('/query')
async def query(query: QueryInput):
    return rag_chatbot.query_documents(query)

@router.get("/get_all_documents")
def get_all_documents():
    return rag_chatbot.get_all_documents()

@router.post("/delete_all_documents")
async def delete_all_documents():
    return rag_chatbot.delete_all_documents()

@router.post("/add_text_documents")
async def add_text_documents(documents: List[UploadFile]):
    general_documents = []
    for document in documents:
        # Read the file content
        if document.content_type == "application/pdf":
            try:
            
                pdf_content = await document.read()
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
                
                # Extract text from each page
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text()
                
                # Generate a unique document ID
                doc_id = str(uuid.uuid4())
                general_documents.append({"id":doc_id, "text": text, "metadata": {"filename": document.filename}})
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Error processing PDF file: {str(e)}"
                )
        else:
            # Handle non-PDF files as plain text
            content = await document.read()
            text = content.decode("utf-8")
            general_documents.append({"text": text, "metadata": {"filename": document.filename}})
        

    
    return rag_chatbot.add_documents(general_documents)

@router.post("/add_json_documents")
async def add_json_documents(documents: List[UploadFile]):
    general_documents = []
    for document in documents:
        json_data = json.loads(await document.read())

        for json_doc in json_data:
            general_documents.append(json_doc)
    
    return rag_chatbot.add_json_documents(general_documents)

@router.post("/init_conversation")
async def init_conversation(user: NewConversationInput):
    return ConversationModel().insert_conversation(user.user)

@router.get("/get_conversation_history/{id}")
def get_conversation_history(id: str):
    return ConversationHistoryModel().get_conversation_history_by_id(id)

@router.post("/insert_conversation_history")
async def insert_conversation_history(idconversation: int, role: str, message: str):
    return ConversationHistoryModel().insert_conversation_history(idconversation, role, message)