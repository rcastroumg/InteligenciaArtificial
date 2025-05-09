import mysql.connector as msql
#import mysqlx
from .settings import Settings
from typing import List

def convert_camel_case(texto: str) -> str:
    """
        Convierte una cadena en formato snake_case a camelCase.


        Args:
            texto (str): La cadena en formato snake_case a convertir.

        Returns:
            str: La cadena convertida en formato camelCase.
    """

    #palabras = texto.split("_")
    #camel_case = '_'.join(palabra.capitalize() for i, palabra in enumerate(palabras))
    #return camel_case
    return texto


class MySqldb:
    """
        Clase que proporciona métodos para conectar y administrar una conexión a una base de datos SQL Server.
        Es necesario instalar: Microsoft ODBC driver for SQL Server

        https://learn.microsoft.com/en-us/sql/connect/odbc/microsoft-odbc-driver-for-sql-server?view=sql-server-ver16        

        Attributes:
            db_user (str): Nombre de usuario para la conexión a la base de datos.
            db_pass (str): Contraseña para la conexión a la base de datos.
            connection (obj): Conexión activa a la base de datos.
    """

    def __init__(self):
        """
            Inicializa una instancia de OracleDBConnector y configura los atributos necesarios para la conexión.
        """

        settings = Settings()
        #self.db_driver = settings.db_driver
        self.db_host = settings.db_host
        self.db_port = settings.db_port
        self.db_name = settings.db_name
        self.db_user = settings.db_user
        self.db_pass = settings.db_pass
        self.connection = None

    def connect(self):
        """
           Establece una conexión.

           Returns:
               None

           Raises:
               ConnectionError: Si ocurre un error al establecer la conexión.
       """
        try:
            db_config = {
                "host":self.db_host,
                "user":self.db_user,
                "password":self.db_pass,
                "database":self.db_name,
                "port":self.db_port
            }
            self.connection = msql.connect(**db_config)
        except msql.DatabaseError as e:
            raise ConnectionError("Error al establecer la conexión: " + str(e))


    def execute_query(self, query: str):
        try:
            self.connect()
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                if cursor.description:
                    result = []
                    for row in cursor:
                        row_dict = {}
                        for i, col in enumerate(cursor.description):
                            col_name = col[0]
                            col_value = row[i]
                            #if isinstance(col_value, odbc.LOB):
                            #    col_value = col_value.read()
                            col_name_camel_case = convert_camel_case(col_name)
                            row_dict[col_name_camel_case] = col_value
                        result.append(row_dict)
                                      
                    return result
                elif cursor._last_insert_id:
                    return cursor._last_insert_id
                else:
                    return None
        except (msql.DatabaseError, msql.InterfaceError) as e:
            return {"OOPS": str(e)}
        finally:
            self.connection.commit()
            cursor.close()
            self.connection.close()  

    def execute_insert(self, query: str, params:List[dict]):
        try:
            self.connect()
            with self.connection.cursor() as cursor:
                send_params = []
                for param in params:
                    param_name = param["nombre"]
                    param_value = param["valor"]

                    send_params.append(param_value)

                cursor.execute(query,send_params)

                if cursor.description:
                    result = []
                    for row in cursor:
                        row_dict = {}
                        for i, col in enumerate(cursor.description):
                            col_name = col[0]
                            col_value = row[i]
                            #if isinstance(col_value, odbc.LOB):
                            #    col_value = col_value.read()
                            col_name_camel_case = convert_camel_case(col_name)
                            row_dict[col_name_camel_case] = col_value
                        result.append(row_dict)
                                        
                    return result
                else:
                    return cursor._last_insert_id
        except (msql.DatabaseError, msql.InterfaceError) as e:
            return {"OOPS": str(e)}
        finally:
            self.connection.commit()
            cursor.close()
            self.connection.close()