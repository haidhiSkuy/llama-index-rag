import os
from src.db_llm import LLMDB 
from openai import AzureOpenAI
from llama_index.core import PromptTemplate
from llama_index.core.schema import NodeWithScore 
from llama_index.core.vector_stores import VectorStoreQuery 
from src.prompt_template import Template

class Retriever:
    def __init__(self):
        llmdb = LLMDB() 
        self.vector_store = llmdb.get_vector_store()
        self.embed_model = llmdb.get_llm_embedding() 

        self.llm = AzureOpenAI(
            api_key=os.getenv("API_KEY"),  
            api_version=os.getenv("API_VERSION"),
            azure_endpoint=os.getenv("API_ENDPOINT")
        ) 
        self.deployment_name = 'corpu-text-gpt-4o'

    def _get_context_string(self, nodes_list : list[NodeWithScore]):
        context_str = " ".join([i.text for i in nodes_list])
        return context_str

    def get_contexts(self, query : str) -> list[NodeWithScore]: 
        query_embedding = self.embed_model.get_query_embedding(query) 

        query_mode = "default"
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding, 
            similarity_top_k=5, 
            mode=query_mode, 
        )

        query_result = self.vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score)) 
        
        return nodes_with_scores
    
    def execute(self, query : str): 
        nodes = self.get_contexts(query) 
        context_string = self._get_context_string(nodes) 
        prompt = Template.summary_template(context_string) 

        message = [
            {"role":"system", "content":""},
            {"role":"user", "content":prompt}
        ]

        response = self.llm.chat.completions.create(
            model='corpu-text-gpt-4o', 
            messages=message, 
            temperature= 0.4, 
            max_tokens=1000
        ) 

        response = response.choices[0].message.content.replace(' .', '.').strip()
        return response
