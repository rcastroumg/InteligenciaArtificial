import re
from starlette.routing import Route
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from utils.settings import Settings
import routers
import uvicorn

app = FastAPI(
    version="1.0.0",
    title="Servicio para inteligencia artificial",
    description="Servicio para inteligencia artificial",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Agrega los Routers
app.include_router(routers.chatbot.router)

# Validaciones al inicial app
@app.on_event("startup")
async def startup_event():
    try:
        settings = Settings()
    except:
        raise Exception("Error al leer archivo .env")
        
    if settings.save_aws:
        if not settings.config_aws:
            raise Exception("No se ha configurado la ruta para el archivo de configuracion de AWS")
        else:
            try:
                file = open(settings.config_aws)
                file.close()
            except Exception as e:
                raise Exception("EL archivo de configuracion para AWS no existe")

@app.get("/", response_class=HTMLResponse)
async def inicio():
    return """
    <html>
        <head>
            <title>DG API Python</title>
        </head>
        <body>
            <h1>DG API Python</h1>
        </body>
    </html>
    """


# Ignorar el case-sensitive
for route in app.router.routes:
    if isinstance(route, Route):
        route.path_regex = re.compile(route.path_regex.pattern, re.IGNORECASE)

# Agrega prefijo
app.mount("/api",app)

# Para debug
if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8091)