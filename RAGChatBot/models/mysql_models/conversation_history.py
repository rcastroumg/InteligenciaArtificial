from repositories.mysql.mysql_db import MySqldb

class ConversationHistoryModel(MySqldb):
    def __init__(self):
        super().__init__()

    def get_conversation_history_by_id(self, id: str):
        query = f"SELECT * FROM conversation_history WHERE idconversation = '{id}'"
        result = self.execute_query(query)
        return result

    def insert_conversation_history(self, idconversation: int, role: str, message: str):
        query = f"INSERT INTO conversation_history (idconversation,role,message) VALUES ({idconversation},'{role}','{message}')"
        result = self.execute_query(query)
        return result