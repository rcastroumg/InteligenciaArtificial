import json
import uuid
from fastapi import HTTPException
from utils.ChatbotUtil import ChatbotUtil
from utils.ConversationContext import ConversationContext
from utils.settings import Settings
from schemas.ChatbotSchema import DocumentInput, QueryInput, QueryOnlyInput
from prompts.RAGChatbotSysPrompt import RAGChatbotSysPrompt
from typing import List
from qdrant_client.http import models
from qdrant_client import QdrantClient

class RAGChatbotAPI:
    def __init__(self):
        self.qdrant_client:QdrantClient = ChatbotUtil.initQdrant()
        self.collection_name = "documents"
        self.embedding_dim = 1024
        
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dim,
                    distance=models.Distance.COSINE
                )
            )
        else:
            self.qdrant_client.get_collection(self.collection_name)
        self.conversation_context = ConversationContext()
    
    def add_documents(self, documents: List[DocumentInput]):
        """
        Añadir documentos a la base de vectores
        """
        try:
            # Prepare points for Qdrant
            points = []
            for i, doc in enumerate(documents):

                chunks = ChatbotUtil.recursive_splitter(doc["text"], chunk_size=1600, chunk_overlap=350)

                for i, chunk in enumerate(chunks):
                    embedding = ChatbotUtil.get_embedding_function_cohere(chunk)
                    points.append(models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "chunk_index": i,
                            **doc["metadata"]
                        }
                    ))
            
            # Upload points to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            return {"status": "Documentos añadidos exitosamente"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        

    def delete_all_documents(self):
        # self.qdrant_client.delete_collection(self.collection_name)
        self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dim,
                    distance=models.Distance.COSINE
                )
            )

    
    def query_documents(self, query: QueryInput):
        """
        Realizar búsqueda de documentos relevantes y generar respuesta
        """
        try:
            query_embedding = ChatbotUtil.get_embedding_function_cohere(query.query)

            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=Settings().qdrant_top_k,
            )

            retrived_docs = [hit.payload["text"] for hit in search_results]
            retrived_ids = [str(hit.id) for hit in search_results]

            context = "\n---\n".join(retrived_docs)
            
            # Obtener historial de conversación
            conversation_history = self.conversation_context.get_conversation_history(
                query.conversation_id
            )
            
            # Generar respuesta con DeepSeek
            response = ""
            response = ChatbotUtil.invoke_deepseek(
                prompt=query.query, 
                context=context, 
                conversation_history=conversation_history,
                sysprompt=RAGChatbotSysPrompt().system_prompt_english
            )
            
            # Añadir mensajes al historial de conversación
            self.conversation_context.add_message(
                conversation_id=query.conversation_id, 
                role="user", 
                message=query.query
            )
            self.conversation_context.add_message(
                conversation_id=query.conversation_id, 
                role="assistant", 
                message=response
            )
            
            return {
                "conversation_id": query.conversation_id,
                "retrieved_docs": retrived_ids,
                "response": response
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    def get_all_collections(self):
        return self.qdrant_client.get_collections()
    


    def only_query_documents(self, query: QueryOnlyInput):
        """
        Realizar búsqueda de documentos relevantes y generar respuesta
        """
        try:
            query_embedding = ChatbotUtil.get_embedding_function_cohere(query.query)

            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=query.top_k,
            )

            retrived_docs = [hit.payload["text"] for hit in search_results]
            retrived_ids = [str(hit.id) for hit in search_results]

            context = "\n---\n".join(retrived_docs)
            
            # Generar respuesta con Claude 3
            # response = ""
            # response = ChatbotUtil.invoke_claude(
            #     prompt=query.query, 
            #     context=context,
            #     sysprompt=query.prompt
            # )
            # 
            # return json.loads(response)
            return context
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))