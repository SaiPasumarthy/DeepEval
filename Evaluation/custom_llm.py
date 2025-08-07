from langchain_openai import AzureChatOpenAI, ChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI

class AzureOpenAI(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "o4-mini"


class CustomModel():
    def __init__(
        self,
        api_key
    ):
        self.api_key = api_key

    def invoke(self) -> AzureOpenAI:
        custom_model = ChatOpenAI(
            model="o4-mini",
            temperature=1,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=self.api_key
        )
        return AzureOpenAI(model=custom_model)
    
class CustomGeminiModel():
    def __init__(
        self,
        api_key
    ):
        self.api_key = api_key

    def invoke(self) -> AzureOpenAI:
        custom_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key
        )
        return AzureOpenAI(model=custom_model)