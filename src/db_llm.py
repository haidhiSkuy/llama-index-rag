import os 
from dotenv import load_dotenv
from llama_index.vector_stores.postgres import PGVectorStore 
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

load_dotenv()


class LLMDB: 
    def __init__(self):
        self.POSTGRES_HOST = os.getenv("POSTGRES_HOST") 
        self.POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
        self.POSTGRES_PORT = os.getenv("POSTGRES_PORT")
        self.POSTGRES_USERNAME = os.getenv("POSTGRES_USERNAME")

        self.OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        self.OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

    def get_vector_store(self) -> PGVectorStore: 
        vector_store = PGVectorStore.from_params(
            database="marshall",
            host=self.POSTGRES_HOST,
            password=self.POSTGRES_PASSWORD,
            port=self.POSTGRES_PORT,
            user=self.POSTGRES_USERNAME,
            table_name="reference_vector_store",
            embed_dim=3072
        )

        return vector_store
    
    def get_llm_embedding(self) -> AzureOpenAIEmbedding:
        embed_model = AzureOpenAIEmbedding(
            model="text-embedding-3-large",
            deployment_name="corpu2-text-embedding-3-large",
            api_key=self.OPENAI_API_KEY,  
            api_version="2024-02-01",
            azure_endpoint=self.OPENAI_ENDPOINT
        )
        return embed_model