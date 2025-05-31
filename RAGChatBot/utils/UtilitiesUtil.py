

import base64
from configparser import ConfigParser
import json

from fastapi import HTTPException, status
from utils.settings import Settings
from prompts.ReadInvoiceSysPrompt import ReadInvoiceSysPrompt
from prompts.TextTransformSysPrompt import TextTransformSysPrompt
from prompts.ValidatePictureSysPrompt import ValidatePictureSysPrompt
from prompts.ReadDocumentSysPrompt import ReadDocumentSysPrompt
from boto3.session import Session

from utils.RAGChatbotAPI import RAGChatbotAPI
from schemas.ChatbotSchema import QueryOnlyInput


config = ConfigParser()

class UtilitiesUtil:
    # Lee las variables de configuracion de aws
    def initAws(client:str):
        config.read(Settings().config_aws)

        AWS_ACCESS_KEY_ID = config.get("aws", "AWS_ACCESS_KEY_ID")
        AWS_SECRET_ACCESS_KEY = config.get("aws", "AWS_SECRET_ACCESS_KEY")
        AWS_REGION = config.get("aws", "AWS_REGION")
        AWS_BUCKET_NAME = config.get("aws", "AWS_BUCKET_NAME")

        session = Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )

        aws_client = session.client(client)

        return [aws_client,AWS_BUCKET_NAME]


    def readPDF(doc_bytes:bytes,prompt:str):
        bedrock,bucket = UtilitiesUtil.initAws("bedrock-runtime")

        doc_message = {
            "role": "user",
            "content": [
                {
                    "document": {
                        "name": "Document 1",
                        "format": "pdf",
                        "source": {
                            "bytes": doc_bytes #Look Ma, no base64 encoding!
                        }
                    }
                },
                {
                    "text": prompt
                }
            ]
        }
        response = bedrock.converse(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            #modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            #modelId="anthropic.claude-3-haiku-20240307-v1:0",
            #modelId="amazon.titan-embed-text-v1",
            messages=[doc_message],
            inferenceConfig={
                "maxTokens": 2000,
                "temperature": 0
            },
        )

        response_text = response['output']['message']['content'][0]['text']

        try:
            return json.loads(response_text)
        except Exception as e:
            raise HTTPException(status_code=400, detail="No se pudo procesar el archivo")



    def readIMG(doc_bytes:bytes,ext:str,prompt:str):
        bedrock,bucket = UtilitiesUtil.initAws("bedrock-runtime")

        doc_b64 = base64.b64encode(doc_bytes).decode()
        doc_prompts = []
        text_prompts = []
        doc_prompts.append({"type": "image", "source": {"type": "base64","media_type": f"image/{ext}","data": doc_b64}})
        text_prompts.append( {"type": "text", "text": prompt})

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096, 
            "temperature": 1.0, 
            "messages": [{
                "role": "user", 
                "content": text_prompts + doc_prompts
            }]
        })


        accept = "application/json"
        contentType = "application/json"
        response = bedrock.invoke_model(
            body=body, 
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0", 
            accept=accept, 
            contentType=contentType
        )
        response_body = json.loads(response.get("body").read())
        try:
            return json.loads(response_body.get("content")[0]["text"])
        except Exception as e:
            raise HTTPException(status_code=400, detail="No se pudo procesar el archivo")
        
    

    def readInvoice(file_content:bytes,ext:str):

        prompt = ReadInvoiceSysPrompt().system_prompt_spanish
        prompt_english = ReadInvoiceSysPrompt().system_prompt_english

        try:
            if ext == "pdf": 
                return UtilitiesUtil.readPDF(file_content,prompt_english)
            elif ext in ["png","jpg","jpeg","gif","webp"]:
                if ext == "jpg":
                    ext = "jpeg"
                return UtilitiesUtil.readIMG(file_content,ext,prompt_english)
            else:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Formato no soportado")
        except Exception as err:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err))
        

    def validatePicture(file_content:bytes,ext:str):
        prompt_english = ValidatePictureSysPrompt().system_prompt_english

        try:
            if ext in ["png","jpg","jpeg"]:
                if ext == "jpg":
                    ext = "jpeg"
                return UtilitiesUtil.readIMG(file_content,ext,prompt_english)
            else:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Formato no soportado")
        except Exception as err:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err))
        

    def textTransform(text:str,type:str,format:str):
        bedrock,bucket = UtilitiesUtil.initAws("bedrock-runtime")        

        type = type.upper()
        format = format.upper()
        temperature = 0.5
        max_tokens = 2000
        prompt = TextTransformSysPrompt().system_prompt_english

        if type == "FORMALIZAR":
            temperature = 0.5
            max_tokens = 2000
        elif type == "DESARROLLAR":
            temperature = 0.8
            max_tokens = 5000
        elif type == "ACORTAR":
            temperature = 0.2
            max_tokens = 1000
        else:
            temperature = 0.5
            max_tokens = 2000

        if format == "HTML":
            prompt = TextTransformSysPrompt().system_prompt_english
            prompt = prompt.replace("{type}",type)
            prompt = prompt.replace("{format}",format)
        elif format == "WHATSAPP":
            prompt = TextTransformSysPrompt().system_prompt_whatsapp_english
            prompt = prompt.replace("{type}",type)
        else:
            prompt = TextTransformSysPrompt().system_prompt_english
            prompt = prompt.replace("{type}",type)
            prompt = prompt.replace("{format}",format)

        doc_message = {
            "role": "user",
            "content": [
                {
                    "text": text
                }
            ]
        }
        sys_message = {
            "text": prompt
        }
        response = bedrock.converse(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            messages=[doc_message],
            system=[sys_message],
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature
            },
        )

        response_text = response['output']['message']['content'][0]['text']

        return response_text
    



    def readWord(doc_bytes:bytes,prompt:str):
        bedrock,bucket = UtilitiesUtil.initAws("bedrock-runtime")

        doc_message = {
            "role": "user",
            "content": [
                {
                    "document": {
                        "name": "Document 1",
                        "format": "docx",
                        "source": {
                            "bytes": doc_bytes #Look Ma, no base64 encoding!
                        }
                    }
                },
                {
                    "text": prompt
                }
            ]
        }
        response = bedrock.converse(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            #modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            #modelId="anthropic.claude-3-haiku-20240307-v1:0",
            #modelId="amazon.titan-embed-text-v1",
            messages=[doc_message],
            inferenceConfig={
                "maxTokens": 2000,
                "temperature": 0
            },
        )

        response_text = response['output']['message']['content'][0]['text']

        try:
            return json.loads(response_text)
        except Exception as e:
            raise HTTPException(status_code=400, detail="No se pudo procesar el archivo")

    def readDocument(file_content:bytes,ext:str):

        prompt = ReadDocumentSysPrompt().system_prompt_spanish
        prompt_english = ReadDocumentSysPrompt().system_prompt_english

        try:
            if ext in ["docx"]: 
                rag_chatbot = RAGChatbotAPI()
                query: QueryOnlyInput = QueryOnlyInput(
                    query=f"""listado de leyes y articulos""",
                    prompt="",
                    top_k=10
                )
                contexto = rag_chatbot.only_query_documents(query)
                prompt_english = prompt_english.replace("{context}",contexto)
                
                datos = UtilitiesUtil.readWord(file_content,prompt_english)
                # for dato in datos:
                #     query: QueryOnlyInput = QueryOnlyInput(
                #         query=f"""Verifica si la siguiente ley está vigente o ha sido reemplazada por otra ley.
                # Ley: {dato["name"]} articulo: {dato["article"]} descripcion: {dato["description"]}""",
                #         prompt="""Por favor, responde a la siguiente pregunta utilizando la información proporcionada,
                #         el formato de la respuesta debe ser un JSON con los siguientes campos:
                #         {
                #             "isvalid": bool,  # Indica si la ley aun está vigente
                #             "replaced_by": str,  # Indica si la ley ha sido reemplazada por otra
                #         }
                #         """,
                #         top_k=10
                #     )
                #     response = rag_chatbot.only_query_documents(query)
                #     dato["isvalid"] = response["isvalid"]
                #     dato["replaced_by"] = response["replaced_by"]
                return datos
            else:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Formato no soportado")
        except Exception as err:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err))