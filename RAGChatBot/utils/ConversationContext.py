from models.mysql_models.conversation_history import ConversationHistoryModel

# Clase para gestionar el contexto de conversación
class ConversationContext:
    def __init__(self, max_history=5):
        """
        Inicializa el gestor de contexto de conversación
        
        :param max_history: Número máximo de mensajes a mantener en el historial
        """
        self.conversations = {}
        self.max_history = max_history
    
    def add_message(self, conversation_id: str, role: str, message: str):
        """
        Añade un mensaje al historial de una conversación
        
        :param conversation_id: ID único de la conversación
        :param role: Rol del mensaje (user/assistant)
        :param message: Contenido del mensaje
        """

        # Guardar historial de conversación en base de datos
        ConversationHistoryModel().insert_conversation_history(conversation_id, role, message)

        # if conversation_id not in self.conversations:
        #     self.conversations[conversation_id] = []
        
        # Añadir mensaje al historial
        # self.conversations[conversation_id].append({
        #     "role": role,
        #     "message": message
        # })
        
        # Limitar el tamaño del historial
        # if len(self.conversations[conversation_id]) > self.max_history * 2:
        #     self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_history*2:]
    
    def get_conversation_history(self, conversation_id: str) -> str:
        """
        Obtiene el historial de una conversación como texto
        
        :param conversation_id: ID único de la conversación
        :return: Historial de conversación formateado
        """

        # Obtener historial de conversación de la base de datos
        conversation_history = ConversationHistoryModel().get_conversation_history_by_id(conversation_id)

        if len(conversation_history) > 0:
            history = []
            for entry in conversation_history:
                history.append(f"{entry['role'].upper()}: {entry['message']}")
            return "\n".join(history)
        else:
            return ""

        # if conversation_id not in self.conversations:
        #     return ""
        
        # Formatear historial como texto
        # history = []
        # for entry in self.conversations[conversation_id]:
        #     history.append(f"{entry['role'].upper()}: {entry['message']}")
        
        #return "\n".join(history)