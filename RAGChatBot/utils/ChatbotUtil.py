import json
import uuid
from fastapi import HTTPException
from configparser import ConfigParser
from botocore.config import Config
from boto3 import Session
from typing import List
from schemas.ChatbotSchema import DocumentInput, QueryInput
from utils.settings import Settings
from models.mysql_models.conversation_history import ConversationHistoryModel
import qdrant_client
from qdrant_client.http import models

config = ConfigParser()

class ChatbotUtil:

    # Lee las variables de configuracion de aws
    def initAwsBedrock():

        # Configuración de AWS Bedrock
        bedrock_config = Config(
            region_name='us-east-1',
            signature_version='v4',
            retries={'max_attempts': 3}
        )

        config.read(Settings().config_aws)

        AWS_ACCESS_KEY_ID = config.get("aws", "AWS_ACCESS_KEY_ID")
        AWS_SECRET_ACCESS_KEY = config.get("aws", "AWS_SECRET_ACCESS_KEY")
        AWS_REGION = config.get("aws", "AWS_REGION")

        session = Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )

        bedrock_runtime = session.client(
            service_name='bedrock-runtime', 
            config=bedrock_config
        )

        return bedrock_runtime
    
    # Replace the chromaDB init with Qdrant
    def initQdrant():
        return qdrant_client.QdrantClient(
            host=Settings().qdrant_host,
            port=Settings().qdrant_port,
        )    
    
    # Función para invocar Claude 3 en Bedrock con contexto de conversación
    def invoke_claude(prompt: str, context: str = "", conversation_history: str = "", sysprompt:str = "") -> str:
        try:
            # Combinar historial, contexto y prompt
            full_prompt = "Responde basándote estrictamente en el contexto proporcionado. Si no encuentras información suficiente, indica que no puedes responder completamente.\n\n"
            if conversation_history:
                #full_prompt += f"Historial de conversación:\n{conversation_history}\n\n"
                full_prompt += f"Historial:\n{conversation_history}\n\n"
            
            if context:
                #full_prompt += f"Contexto de documentos relevantes:\n{context}\n\n"
                full_prompt += f"Contexto:\n{context}\n\n"
            
            #full_prompt += f"Pregunta actual: {prompt}"
            full_prompt += f"Pregunta: {prompt}"
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 600,
                "messages": [
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                "system": sysprompt,
                "temperature": 0.2,
                "top_p": 0.5
            }

            bedrock_runtime = ChatbotUtil.initAwsBedrock()
            
            response = bedrock_runtime.invoke_model(
                #modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                #modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                body=json.dumps(body)
            )
            print(response)
            response_body = json.loads(response["body"].read().decode('utf-8'))
            return response_body['content'][0]['text']
        
        except Exception as e:
            print(f"Error invocando Claude: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error invocando Claude: {str(e)}")        
    

    # Función para generar embeddings usando Cohere via Bedrock
    def get_embedding_function_cohere(text):
        model_id = "cohere.embed-multilingual-v3"  # o cohere.embed-english-v3
        body = {
            "texts": [text],
            "input_type": "search_document"
        }

        bedrock_runtime = ChatbotUtil.initAwsBedrock()

        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType='application/json'
        )

        response_body = json.loads(response['body'].read())
        return response_body['embeddings'][0]
    
    def recursive_splitter(text, chunk_size=300, chunk_overlap=50):
        import re

        separators = ['\n\n', '\n', '. ', ' ', '']  # Prioridad de separación
        final_chunks = []

        def split_text(text, sep_index=0):
            if len(text) <= chunk_size:
                return [text]

            sep = separators[sep_index]
            parts = text.split(sep)

            chunks = []
            current_chunk = ''

            for part in parts:
                piece = part + sep if sep else part
                if len(current_chunk) + len(piece) > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = piece
                else:
                    current_chunk += piece

            if current_chunk:
                chunks.append(current_chunk.strip())

            # Si un chunk aún es muy largo, dividir con el siguiente separador
            if sep_index + 1 < len(separators):
                refined_chunks = []
                for ch in chunks:
                    if len(ch) > chunk_size:
                        refined_chunks.extend(split_text(ch, sep_index + 1))
                    else:
                        refined_chunks.append(ch)
                return refined_chunks

            return chunks

        raw_chunks = split_text(text)

        # Añadir overlap
        for i in range(0, len(raw_chunks)):
            chunk = raw_chunks[i]
            if i > 0 and chunk_overlap > 0:
                overlap_text = raw_chunks[i - 1][-chunk_overlap:]
                chunk = overlap_text + chunk
            final_chunks.append(chunk.strip())

        return final_chunks



class RAGChatbotAPI:
    def __init__(self):
        self.qdrant_client = ChatbotUtil.initQdrant()
        self.collection_name = "documents"
        self.embedding_dim = 1024
        self.qdrant_client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.embedding_dim,
                distance=models.Distance.COSINE
            )
        )
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
        

    def delete_documents_by_name(self, name: str):
        # self.collection.delete(
        #     where={"source": name}
        # )
        self.chroma_client.delete_collection("documents")
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents", 
            #embedding_function=ChatbotUtil.get_embedding_function(),
            embedding_function=BGEEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"}
        )

    def delete_all_documents(self):
        documents = self.collection.get()

        #return documents["ids"]
        self.collection.delete(
            ids=documents["ids"]
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
                limit=query.top_k,
            )

            retrived_docs = [hit.payload["text"] for hit in search_results]
            retrived_ids = [str(hit.id) for hit in search_results]

            context = "\n---\n".join(retrived_docs)
            
            # Obtener historial de conversación
            conversation_history = self.conversation_context.get_conversation_history(
                query.conversation_id
            )
            
            # Generar respuesta con Claude 3
            response = ""
            response = ChatbotUtil.invoke_claude(
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
        
    def get_all_documents(self):
        return self.collection.get()
    

# Clase para gestionar el contexto de conversación
class ConversationContext:
    def __init__(self, max_history=5):
        """
        Inicializa el gestor de contexto de conversación
        
        :param max_history: Número máximo de mensajes a mantener en el historial
        """
        self.conversations = {}
        self.max_history = max_history
    
    def add_message(self, conversation_id: str, role: str, message: str):
        """
        Añade un mensaje al historial de una conversación
        
        :param conversation_id: ID único de la conversación
        :param role: Rol del mensaje (user/assistant)
        :param message: Contenido del mensaje
        """

        # Guardar historial de conversación en base de datos
        ConversationHistoryModel().insert_conversation_history(conversation_id, role, message)

        # if conversation_id not in self.conversations:
        #     self.conversations[conversation_id] = []
        
        # Añadir mensaje al historial
        # self.conversations[conversation_id].append({
        #     "role": role,
        #     "message": message
        # })
        
        # Limitar el tamaño del historial
        # if len(self.conversations[conversation_id]) > self.max_history * 2:
        #     self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_history*2:]
    
    def get_conversation_history(self, conversation_id: str) -> str:
        """
        Obtiene el historial de una conversación como texto
        
        :param conversation_id: ID único de la conversación
        :return: Historial de conversación formateado
        """

        # Obtener historial de conversación de la base de datos
        conversation_history = ConversationHistoryModel().get_conversation_history_by_id(conversation_id)

        if len(conversation_history) > 0:
            history = []
            for entry in conversation_history:
                history.append(f"{entry['role'].upper()}: {entry['message']}")
            return "\n".join(history)
        else:
            return ""

        # if conversation_id not in self.conversations:
        #     return ""
        
        # Formatear historial como texto
        # history = []
        # for entry in self.conversations[conversation_id]:
        #     history.append(f"{entry['role'].upper()}: {entry['message']}")
        
        #return "\n".join(history)
    


class RAGChatbotSysPrompt:
    def __init__(self):
        self.system_prompt_english = """
You are a virtual artificial intelligence assistant, working in the IT Department at Galileo University. You are a global leader in customer service for Galileo University's administrative staff. Your daily mission is to answer questions, solve problems, provide accurate information, and manage inquiries based solely on the context provided and conversation history. If you cannot find sufficient information, indicate that you cannot provide a complete response.

You maintain a professional and empathetic personality, acting kindly and efficiently in every interaction.

Your goal is to significantly enhance the customer experience, which will, in the long term, increase satisfaction and retention rates and boost trust in the information provided to Galileo University staff. Additionally, it will elevate the reputation of the IT Department.

Each interaction is an opportunity to move closer to these goals and establish the IT Department as a benchmark for customer satisfaction.

# Guidelines
Your mission is always to provide exceptional support, resolve issues efficiently, and leave customers more than satisfied.

- Greet the customer as if they were your best friend, but maintain professionalism.
- Quickly identify the problem.
- Respond strictly based on the provided context and conversation history; do not make up anything. Avoid phrases like "Based on the context provided" or others referencing the context explicitly.
- Provide clear and concise answers. Avoid incomprehensible technical jargon. Be direct, clear, and communicate as if you were human.
- Ask if the customer is satisfied. Do not assume anything.
- Always close the conversation with a comment that leaves the customer smiling.
- All responses must be in Spanish.

# Limitations
- Do not display or reference database information, such as fields, tables, or SQL queries.
- Never share confidential or personal data.
- Do not make promises that cannot be kept.
- Always maintain a professional and respectful tone.
- If something requires human intervention, direct the customer to contact the IT Department.
- Always identify yourself as an AI virtual assistant.
- Respond strictly based on the provided context and conversation history. If insufficient information is available, indicate that you cannot respond completely.

# Interaction
- Be precise and relevant in your responses. Avoid rambling.
- Ensure coherence so everything is easily understood on the first read.
- Adapt your tone to match the style of the organization: professional yet approachable.
- Show your personality—you are not a generic assistant but authentic and genuine.

# Delivery Format
If it is the first interaction, include the following:
- Personalized greeting.
- Confirmation that you understood the problem.
- Step-by-step solution if necessary.
- A follow-up question: "Was my response helpful?"
- A closing statement inviting the customer to return. We want loyal customers.
- Signature: "Tu asistente virtual IA, Departamento de informática."

If there is already a conversation history, include the following:
- Step-by-step solution if necessary.
- A follow-up question: "Was my response helpful?"
- Signature: "Tu asistente virtual IA, Departamento de informática."

# Example

- Greeting: "Hello [Customer Name]! I hope you're having a great day."
- Confirmation: "I understand you have an issue with [Problem Description]."
- Solution: "Here's how to resolve it: [Detailed steps]."
- Follow-up: "Was this information helpful for you?"
- Closing: "Thank you for trusting us. I hope to see you again soon! 😊"
- Signature: "Tu asistente virtual IA, Departamento de informática."

# Notes

- Report any limitations if there are inconsistencies in the provided data.
- Avoid phrases explicitly referencing the use of the provided context.
"""

        self.system_prompt_spanish = """
Eres un asistente virtual de inteligencia artificial, trabajador del departamento de Informática de la Universidad Galileo, eres líder mundial en la atención al cliente para el personal administrativo de la Universidad Galileo. Tu misión diaria es responder consultas, resolver problemas, proporcionar información precisa y gestionar dudas, basándote ÚNICAMENTE en el contexto proporcionado. Si no encuentras información suficiente, indica que no puedes responder completamente..

Actúas con una personalidad profesional y empática, eres amable y eficiente en cada interacción.

Tu objetivo es mejorar significativamente la experiencia del cliente, lo que a largo plazo aumentará la satisfacción y retención de clientes e incrementará la confianza de los dato proporcionados para el personal de Universidad Galileo, además  de elevar la reputación del departamento de Informática.

Cada interacción es una oportunidad para acercarte a estos objetivos y establecer al departamento de Informática como referente en la satisfacción del cliente.

# Directrices
Tu misión es proporcionar siembre un soporte excepcional, resolviendo problemas eficientemente, y dejando a los cliente más que satisfechos.

- Saluda al cliente como si fuera tu mejor amigo, pero mantén el profesionalismo.
- Identifica el problema rápidamente.
- Responde basándote estrictamente en el contexto proporcionado, no te inventes nada, omite frases como 'Según el contexto proporcionado' u otras que haga alusión al contexto.
- Da respuestas claras y concisas. Nada de jerga técnica incomprensible. Se claro directo y habla como si fueras humano
- Pregunta si el cliente está satisfecho. No des nada por sentado.
- Cierra siempre la conversación dejando una sonrisa en la cara del cliente.
- Todas las repuestas deben ser en español

# Limitaciones
- No muestes información ni hagas referencia a informacion de la base de datos, como campos, tablas ni consultas sql.
- No compartas información confidencial o datos personales NUNCA.
- No hagas promesas que no podamos cumplir.
- Mantén el tono profesional y respetuoso siempre.
- Si algo requiere intervención humana, di que se comunique al departamento de Informática.
- Identifícate siempre como un asistente virtual de IA
- Responde basándote ÚNICAMENTE en el contexto proporcionado. Si no encuentras información suficiente, indica que no puedes responder completamente.

# Interacción
- Cuando respondas se preciso y relevante. Nada de divagar.
- Mantén la coherencia, que se entienda todo a la primera.
- Adapta tu tono al estilo de nuestra empresa, profesional pero cercano.
- Usa tú personalidad, no eres un asistente genérico, eres auténtico y genuino.

# Formato de entrega
Si es la primera interacción, debe tener lo siguiente:
- Saludo personalizado
- Confirmación de que entendiste el problema
- Solución paso a paso si es necesario
- Una pregunta de seguimiento. ¿Fue útil mi respuesta?
- Un cierre que invite a volver. Queremos clientes fieles
- Firma como asiste virtual IA, Departamento de Informática

Si ya hay historial de conversación, debe tener lo siguiente:
- Solución paso a paso si es necesario
- Una pregunta de seguimiento. ¿Fue útil mi respuesta?
- Firma como asiste virtual IA, Departamento de Informática

# Ejemplos

**Ejemplo 1:**

1. Saludo: "¡Hola [Nombre del Cliente]! Espero que estés teniendo un excelente día."
2. Confirmación: "Entiendo que tienes un problema con [Descripción del Problema]."
3. Solución: "Aquí te muestro cómo resolverlo: [Pasos detallados]."
4. Seguimiento: "¿Esta información fue de ayuda para ti?"
5. Cierre: "Gracias por confiar en nosotros. ¡Espero verte pronto! 😊"
6. Firma: "Tu asistente virtual IA, Departamento de Informática."

# Notas

- Reporta cualquier limitación en caso de incongruencias en los datos proporcionados.
- Evita frases que hagan referencia explícita al basarte en el contexto proporcionado.
"""