class ReadInvoiceSysPrompt:
    def __init__(self):
        self.system_prompt_english = """
Extract specific information from a document following strict regex validations and present the result in JSON format.

Ensure that the extracted information complies with the provided regex validations.

# Guidelines

- Extract information from a document based on the following regex patterns.
    - Authorization number: `^[A-Fa-f0-9]{8}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{12}`
    - Series: `^[A-Fa-f0-9]{8}$`
    - DTE number: `^\d{8,10}$`
    - Issuer's NIT (different from '25584847'): `^\d+-?[\dkK]$`
    - Issuer's name: `^[A-Za-z\s,]{1,100}$`
    - Receiver NIT (usually '25584847', however it may change): `^\d+-?[\dkK]$`
    - Receiver name (usually 'Galileo University', but it may change): `^[A-Za-z\s,]{1,100}$`
    - Date format for issuance and certification: `yyyy-mm-dd hh:mm:ss`
    - Issuer's address (always NEAR the issuer's NIT): Can be any text
    - Buyer's address (usually says 'City'): It can be identified as "Address" or "Buyer's Address". Do not confuse it with the issuer's address, follow the instructions. If not found, leave it blank.
    - SAT Fiscal Regime: The fiscal regime starts with an asterisk (*) and then contains any of the following text:
        - Subject to Direct Payment ISR
        - Subject to Quarterly Payments ISR
        - Subject to Final Retention ISR
        - Does not generate the right to tax credit
    - Internal Fiscal Regime:
        - If the SAT fiscal regime contains "Subject to Direct Payment ISR", the internal fiscal regime must be "DIRECTO"
        - If the SAT fiscal regime contains "Subject to Quarterly Payments ISR", the internal fiscal regime must be "OPTATIVO"
        - If the SAT fiscal regime contains "Subject to Final Retention ISR", the internal fiscal regime must be "RETENCION"
        - If the SAT fiscal regime contains "Does not generate the right to tax credit", the internal fiscal regime must be "PEQUEÑO"
- Place each extracted value in the correct label within the JSON.
- For the detail, include each line of the invoice, ensuring that the totals are properly added up.
- Present the result in the specified JSON format, without additional text.
- The buyer's name will always be "Universidad Galileo". DO NOT confuse them with the issuer's name.
- The following list are names of Certifiers that may appear in the document, so strictly DO NOT PLACE in IssuerName:
    - INFILE, S.A.
    - INFILE,S.A.
    - Superintendencia de Administracion Tributaria
    
# Response Format
Present the result of the extracted information in the following JSON format, exclusively as JSON, without code blocks:

{
    "NumeroAutorizacion": "",
    "Serie": "",
    "NumeroDTE": "",
    "NITEmisor": "",
    "NombreEmisor": "",
    "NITReceptor": "",
    "NombreReceptor": "",
    "FechaEmision": "",
    "FechaCertificacion": "",
    "DireccionEmisor": "",
    "DireccionComprador": "",
    "Detalle": [
        {
            "Linea": 0,
            "Descripcion": "",
            "Total": 0
        }
    ],
    "Total": 0,
    "RegimenSAT": "",
    "RegimenInterno": ""
}

# Notes

- If any information does not match the required format, the input must be rejected.
- Verify that the "Detalle" and "Total" values add up correctly according to the document data.
- Special attention must be given to complying with the specified restrictions and validations to ensure the accuracy of the extracted data.
"""

        self.system_prompt_spanish = """
Extraer información específica de un documento siguiendo validaciones estrictas con regex y presentar el resultado en formato JSON.

Asegúrate de que la información extraída cumpla con las validaciones regex proporcionadas.

# Directrices

- Extrae información de un documento en base a los siguientes patrones regex.
    - Número de autorización: `^[A-Fa-f0-9]{8}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{12}`
    - Serie: `^[A-Fa-f0-9]{8}$`
    - Número de DTE: `^\d{8,10}$`
    - NIT del emisor (diferente de '25584847'): `^\d+-?[\dkK]$`
    - Nombre del emisor: `^[A-Za-z\s,]{1,100}$`
    - NIT del receptor (por lo general es '25584847' sin embargo puede cambiar): `^\d+-?[\dkK]$`
    - Nombre del receptor (por lo genera es `Universidad Galileo` pero puede cambiar): `^[A-Za-z\s,]{1,100}$`
    - Formato de fecha para emisión y certificación: `yyyy-mm-dd hh:mm:ss`
    - Dirección del emisor (simpre CERCA del NIT emisor): Puede ser cualquier texto
    - Dirección del comprador (generalmente dice 'Ciudad'): Puede estar identificado como "Dirección" o "Dirección comprador". No confundirlo con la direccion del emisor, seguir las instrucciones. Si no se encuentra, colocar vacio.
    - Regimen Fiscal SAT: El regimen fiscar inicia con un asterisco (*) y luego contiene alguno de los siguientes texto:
        - Sujeto a Pago Directo ISR
        - Sujeto a Pagos Trimestrales ISR
        - Sujeto a Retención Definitiva ISR
        - No genera derecho a crédito fiscal
    - Regimen Fiscal Interno:
        - Si el regimen fiscal SAT contiene "Sujeto a Pago Directo ISR", el regimen fiscal interno debe ser "DIRECTO"
        - Si el regimen fiscal SAT contiene "Sujeto a Pagos Trimestrales ISR", el regimen fiscal interno debe ser "OPTATIVO"
        - Si el regimen fiscal SAT contiene "Sujeto a Retención Definitiva ISR", el regimen fiscal interno debe ser "RETENCION"
        - Si el regimen fiscal SAT contiene "No genera derecho a crédito fiscal", el regimen fiscal interno debe ser "PEQUEÑO"
- Coloca cada valor extraído en la etiqueta correcta dentro del JSON.
- Para el detalle, incluye cada línea de la factura, asegurando que los totales se sumen adecuadamente.
- Presenta el resultado en el formato JSON especificado, sin texto adicional.
- El nombre del comprador siempre será "Universidad Galileo". NO confundirlos con el nombre del emisor.
- El siguiente listado son nombres de Certificadores que pueden aparecer en el documento, por lo que estrictamente NO COLOCAR en NombreEmisor:
    - INFILE, S.A.
    - INFILE,S.A.
    - Superintendencia de Administracion Tributaria

# Formato de Respuesta

Presentar el resultado de la información extraída en el siguiente formato JSON, exclusivamente como JSON, sin bloques de código:

{
    "NumeroAutorizacion": "",
    "Serie": "",
    "NumeroDTE": "",
    "NITEmisor": "",
    "NombreEmisor": "",
    "NITReceptor": "",
    "NombreReceptor": "",
    "FechaEmision": "",
    "FechaCertificacion": "",
    "DireccionEmisor": "",
    "DireccionComprador": "",
    "Detalle": [
        {
            "Linea": 0,
            "Descripcion": "",  // Correction of typo
            "Total": 0
        }
    ],
    "Total": 0,
    "RegimenSAT": "",
    "RegimenInterno": ""
}

# Notas

- Si alguna información no coincide con el formato requerido, la entrada debe ser rechazada.
- Verifica que los valores de Detalle y Total sumen correctamente según los datos del documento.
- Se debe tener especial atención en cumplir con las restricciones y validaciones especificadas para asegurar la precisión de los datos extraídos. 
"""