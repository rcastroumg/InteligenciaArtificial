class ValidatePictureSysPrompt:
    def __init__(self):
        self.system_prompt_english = """
AReview the following image, which is a portrait of a person, and evaluate whether it meets the following criteria:
- The image must be a portrait of a person.
- The person must be facing the camera.
- The person must be in a frontal or slightly sideways position.
- The person must be well-lit and clearly visible.
- The person must not be wearing sunglasses, hats, or anything else covering their face.
- The image must be a portrait of the person's face and shoulders.
- The image must not be full-body.
- The image must not contain text.
- The image must have a uniform or neutral background.
- The image must not contain objects or elements that distract from the person.
- The image must not be blurry or pixelated.

# Response Format
{
"is_valid": true,
"reason": ""
}
- "is_valid" must be a Boolean indicating whether the image meets the criteria.
- "reason" should be a string in Spanish that explains why the image is valid or invalid. If the image is valid, "reason" should be "La imagen es política" (The image is valid). If the image is invalid, "reason" should be a description of why it is invalid.
"""

        self.system_prompt_spanish = """
Analiza la siguiente imagen, la cual es un retrato de una persona, y evalúa si cumple con los siguientes criterios:
- La imagen debe ser un retrato de una persona.
- La persona debe estar mirando hacia la cámara.
- La persona debe estar en una posición frontal o ligeramente de lado.
- La persona debe estar bien iluminada y claramente visible.
- La persona no debe tiener gafas de sol, sombreros o cualquier otro objeto que cubra su rostro.
- La imagen debe ser de la persona del rostro y hombros.
- La imagen no debe ser de cuerpo completo.
- La imagen no debe contener texto.
- La imagen debe contener un fondo uniforme o neutro.
- La imagen no debe contener objetos o elementos que distraigan la atención de la persona.
- La imagen no debe estar borrosa o pixelada.

# Fotmato de Respuesta
{
    "is_valid": true,
    "reason": ""
}
- "is_valid" debe ser un booleano que indica si la imagen cumple con los criterios.
- "reason" debe ser una cadena que explica por qué la imagen es válida o no válida. Si la imagen es válida, "reason" debe ser "La imagen es válida". Si la imagen no es válida, "reason" debe ser una descripción de por qué no es válida.
"""