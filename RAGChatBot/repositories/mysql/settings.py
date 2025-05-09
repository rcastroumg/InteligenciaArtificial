import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    db_driver:str = os.getenv('DB_DRIVER_MYSQL')
    db_host:str = os.getenv('DB_HOST_MYSQL')
    db_port:str = os.getenv('DB_PORT_MYSQL')
    db_name:str = os.getenv('DB_NAME_MYSQL')
    db_user:str = os.getenv('DB_USER_MYSQL')
    db_pass:str = os.getenv('DB_PASS_MYSQL')