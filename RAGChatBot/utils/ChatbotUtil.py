import json
from fastapi import HTTPException
from configparser import ConfigParser
from botocore.config import Config
from boto3 import Session
from utils.settings import Settings
import qdrant_client


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
    def initQdrant() -> qdrant_client.QdrantClient:
        return qdrant_client.QdrantClient(
            host=Settings().qdrant_host,
            port=Settings().qdrant_port,
        )    
    

    # Función para invocar DeepSeek en Bedrock con contexto de conversación
    def invoke_deepseek(prompt: str, context: str = "", conversation_history: str = "", sysprompt:str = "") -> str:
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
            # Embed the prompt in DeepSeek-R1's instruction format.
            formatted_prompt = f"""
            <｜begin▁of▁sentence｜><｜System｜>{sysprompt}<｜User｜>{full_prompt}
            """
            body = {
                "prompt": formatted_prompt,
                "temperature": 0.0, 
                "top_p": 0.5,
                "max_tokens": 600,
            }

            bedrock_runtime = ChatbotUtil.initAwsBedrock()

            # Construimos el ARN del perfil
            inference_profile_arn = "arn:aws:bedrock:us-west-2:210817648150:inference-profile/us.deepseek.r1-v1:0"

            # Construimos los headers especiales
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Amzn-Bedrock-Inference-Profile-ARN": inference_profile_arn
            }
            
            response = bedrock_runtime.invoke_model(
                #modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                #modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
                #modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                modelId="us.deepseek.r1-v1:0",
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json',
            )
            
            response_body = json.loads(response["body"].read().decode('utf-8'))
            #return response_body['content'][0]['text']
            return response_body['choices'][0]['text']
        
        except Exception as e:
            print(f"Error invocando Claude: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error invocando Claude: {str(e)}")  
    
    # Función para invocar Claude 3 en Bedrock con contexto de conversación
    def invoke_claude(prompt: str, context: str = "", conversation_history: str = "", sysprompt:str = "") -> str:
        try:
            # Combinar historial, contexto y prompt
            full_prompt = "Responde basándote estrictamente en el contexto proporcionado. Responde unicamente en formato JSON porporcionado.\n\n"
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
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json',
            )
            
            response_body = json.loads(response["body"].read().decode('utf-8'))
            return response_body['content'][0]['text']
            #return response_body['choices'][0]['text']
        
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
