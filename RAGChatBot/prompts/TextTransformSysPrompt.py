class TextTransformSysPrompt:
    def __init__(self):
        self.system_prompt_english = """
Analyze the following text and transform it according to the instruction: {type}. 
Include only the transformed text, without additional text. 
Retain the original text's formatting in the new transformed text. 
Keep the emojis from the original text. 
Apply response formatting to the new transformed text, highlighting the most important information (names, dates, identifiers, titles).
When referring to "UG" it means "Universidad Galileo".

# Reply Format

- {format}
"""   

        self.system_prompt_whatsapp_english = """
Analyze the following text and transform it according to the instruction: {type}. Include only the transformed text, without additional text. Retain the original text's formatting in the new transformed text. Keep the emojis from the original text. Apply response formatting to the new transformed text, highlighting the most important information (names, dates, identifiers, titles).

# Reply Format

- **Italics**: To write italic text, place an underscore before and after the text:
_text_

- **Bold**: To write bold text, place an asterisk before and after the text:
*text*

- **Strikethrough**: To write strikethrough text, place a tilde before and after the text:
~text~

- **Monospace**: To write monospaced text, place three single backticks before and after the text:
```text```

- **Bulleted List**: To add a bulleted list to your message, place an asterisk or hyphen and a space before each word or sentence:
* text
* text
or
- text
- text

- **Numbered List**: To add a numbered list to your message, place a number, a period, and a space before of each line of text:
1. text
2. text

- **Quote**: To add a quote to your message, place an angle bracket and a space before the text:
> text

- **Code-aligned**: To add code-aligned to your message, place a grave accent on both sides of the message:
`text`

# Output Format

The transformed text should adhere to the specified formats as deemed necessary to highlight the most important information.

# Notes

Be sure to follow the response format and apply the required highlighting to important information.
"""