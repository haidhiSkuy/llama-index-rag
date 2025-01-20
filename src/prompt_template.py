from llama_index.core import PromptTemplate 

class Template: 
    @staticmethod
    def summary_template(context : str) -> str: 
        template = (
            "We have provided context information below. \n"
            "---------------------\n"
            "{context}"
            "\n---------------------\n"
            "Given this information, please give the summary"
        )
        sum_template = PromptTemplate(template) 
        prompt = sum_template.format(context=context)
        return prompt 