import openai
from openai.types.chat import ChatCompletion

from utils.logger import Logger

class LLM:
    def __init__(self, base_url: str, model: str, api_key: str):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        
        self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
    
    def get_answer(self, question: str, max_tokens: int) -> str | None:
        messages=[
            {"role": "user", "content": question}
        ]

        try:
            response: ChatCompletion = self.client.chat.completions.create(
                messages=messages,  # type: ignore
                model=self.model,
                max_tokens=max_tokens,
                stream=False
            )  # type: ignore
        except Exception:
            Logger.error('get_answer', f'Unable to connect to the OpenAI API at {self.base_url}')
            return None

        try:
            response_text = response.choices[0].message.content.strip()  # type: ignore
        except Exception:
            Logger.error('get_answer', f'The response from the model was invalid (no content)')
            return None
        
        return response_text

