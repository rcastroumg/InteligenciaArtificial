class ReadDocumentSysPrompt:
    def __init__(self):
        self.system_prompt_english = """
Extract specific information from a document, usually legal in nature. This is why all laws, decrees, articles, regulations, agreements, resolutions, among others, mentioned in the document are required, and present it in JSON format.

Additionally, based on the following context, it validates whether the law is still in effect. If it is not, it will indicate that it has been repealed. Set the isActive field to false. If it is, it is because it is NOT in the context. Set the isActive field to true.
Context:
{context}

# Response Format

Present the extracted information in the following JSON format, exclusively as JSON, without code blocks:

[
    {
        id: "", // Unique identifier of the article or law, starting with "law_" followed by a sequential number.
        name: "", // Name of the Law, Decree, Agreement, Resolution, Regulation, etc.
        article: "", // Specific article of the law
        description: // Full description of the law if it exists
        isActive: true, // Fill based on the provided context, random boolean value (false, true)
        lastUpdate: "2022-03-15"
    },
]
"""

        self.system_prompt_spanish = """
Extraer información específica de un documento por lo general de ambito legal, es por ello que se necesitan todas las leyes, decretos, articulos, reglamentos, acuerdos, resoluciones entre otros, que se mensionen en el documento,
y presentarlo en formato JSON.

Ademas en base al siguiente contexto, valida si la ley aun esta vigente, si no lo esta indicará que se encuentra derogado, coloca el campo isActive en false, si lo esta es porque NO esta en el contexto, coloca el campo isActive en true.
Contexto:
{context}

# Formato de Respuesta

Presentar el resultado de la información extraída en el siguiente formato JSON, exclusivamente como JSON, sin bloques de código:

[
    {
        id: "", // Identificador único del artículo o ley, iniciando con "law_" seguido de un número secuencial.
        name: "", // Nombre de la Ley, Decreto, Acuerdo, Resolución, Reglamento, etc.
        article: "", // Artículo específico de la ley
        description:  // Descripcion completa de la ley si existiera
        isActive: true, // Llenar en base al contexto proporcionado, valor booleano aleatorio (false, true)
        lastUpdate: "2022-03-15"
    },
]

"""