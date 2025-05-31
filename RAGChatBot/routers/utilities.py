import base64
from fastapi import APIRouter, UploadFile, HTTPException, status

# from models.pkg_digitalizacion import PkgDigitalizacion
# from schemas.ArchivosSchema import ArchivoResult
# from utils.ArchivosUtil import ArchivosUtil
from schemas.UtilitiesSchema import readInvoiceRequest, textTransformRequest
from utils.UtilitiesUtil import UtilitiesUtil

router = APIRouter(
    prefix='/utilidades',
    tags=['Utilidades'],
)


@router.post('/iaReadInvoice')
async def read_invoice(idArchivo:int):
    # ret = PkgDigitalizacion().Archivo(idArchivo)[0]
    # archivo = ArchivoResult(Ruta=ret["Ruta"],Nombre=ret["Nombre"],Extension=ret["Extension"],Extencionfisica=ret["Extencionfisica"])
    # ext = ret["Extencionfisica"]
    # if not archivo:
    #     raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Archivo no encontrado")
# 
    # try:
    #     # Obtener el archivo
    #     file_content = ArchivosUtil.extraerArchivoFisico(archivo,idArchivo)
    # except IOError:
    #     raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error al leer el archivo")
    # except Exception as err:
    #     raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err))    
# 
    # return UtilitiesUtil.readInvoice(file_content,ext)
    pass 
    

@router.post('/iaReadInvoiceContent')
async def read_invoice_content(model:readInvoiceRequest):
    file_content:bytes = []
    try:
        #string is UTF-8
        base64_code = model.Contenido.decode("utf-8")
        data = base64_code.encode()
        file_content = base64.b64decode(data)
    except UnicodeError:
        #string is not UTF-8
        file_content = model.Contenido 

    ext = model.Ext

    return UtilitiesUtil.readInvoice(file_content,ext)

@router.post('/iaReadInvoiceTesting')
async def read_invoice_testing(Ext:str,Contenido:UploadFile):
    file_content:bytes = []
    try:
        file_content = await Contenido.read()
    except UnicodeError:
        raise HTTPException(status_code=400, detail="No se pudo procesar el archivo")

    ext = Ext

    return UtilitiesUtil.readInvoice(file_content,ext)



@router.post('/iaTextTransform')
async def text_transform(model:textTransformRequest):
    return UtilitiesUtil.textTransform(model.Texto,model.Tipo,model.Formato)


@router.post('/iaValidatePictureContent')
async def validate_picture_testing(model:readInvoiceRequest):
    file_content:bytes = []
    try:
        #string is UTF-8
        base64_code = model.Contenido.decode("utf-8")
        data = base64_code.encode()
        file_content = base64.b64decode(data)
    except UnicodeError:
        #string is not UTF-8
        file_content = model.Contenido 

    ext = model.Ext

    return UtilitiesUtil.validatePicture(file_content,ext)


@router.post('/iaValidatePictureTesting')
async def validate_picture_testing(Contenido:UploadFile):
    file_content:bytes = []
    ext = ""
    try:
        file_content = await Contenido.read()
        ext = Contenido.filename.split(".")[-1]
    except UnicodeError:
        raise HTTPException(status_code=400, detail="No se pudo procesar el archivo")


    return UtilitiesUtil.validatePicture(file_content,ext)




@router.post('/iaReadDocument')
async def read_document(Contenido:UploadFile):
    file_content:bytes = []
    ext = ""
    try:
        file_content = await Contenido.read()
        ext = Contenido.filename.split(".")[-1]
    except UnicodeError:
        raise HTTPException(status_code=400, detail="No se pudo procesar el archivo")


    return UtilitiesUtil.readDocument(file_content,ext)