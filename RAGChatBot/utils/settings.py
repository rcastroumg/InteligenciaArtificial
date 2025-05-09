import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):

    def strtobool (val):
        """Convert a string representation of truth to true (1) or false (0).
        True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
        are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
        'val' is anything else.
        """
        val = val.lower()
        if val in ('y', 'yes', 't', 'true', 'on', '1'):
            return 1
        elif val in ('n', 'no', 'f', 'false', 'off', '0'):
            return 0
        else:
            raise ValueError("invalid truth value %r" % (val,))

    config_aws:str = os.getenv('CONFIG_AWS')

    qdrant_host:str = os.getenv('QDRANT_HOST')
    qdrant_port:int = int(os.getenv('QDRANT_PORT'))
    save_aws:bool = strtobool(os.getenv('SAVE_AWS'))
    
    # chromadb_path:str = os.getenv('CHROMADB_PATH')
    # api_reclutamiento:str = os.getenv('API_RECLUTAMIENTO')
    # token_reclutamiento:str = os.getenv('TOKEN_RECLUTAMIENTO')