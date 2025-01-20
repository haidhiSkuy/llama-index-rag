from src.retriever import Retriever

retriever = Retriever()

query = "cara membuat UI yang menarik"
result = retriever.execute(query)

print(result)