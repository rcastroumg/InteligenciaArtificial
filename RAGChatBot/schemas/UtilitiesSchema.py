from pydantic import BaseModel

class readInvoiceRequest(BaseModel):
    Ext:str
    Contenido:bytes


class textTransformRequest(BaseModel):
    Texto:str
    Tipo:str
    Formato:str