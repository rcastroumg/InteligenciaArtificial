from repositories.mysql.mysql_db import MySqldb

class ConversationModel(MySqldb):
    def __init__(self):
        super().__init__()

    def get_conversation_by_id(self, id: str):
        query = f"SELECT * FROM conversation WHERE idconversation = '{id}'"
        result = self.execute_query(query)
        return result

    def insert_conversation(self, user_id: str):
        query = f"INSERT INTO conversation (user) VALUES ('{user_id}')"
        result = self.execute_query(query)
        return result