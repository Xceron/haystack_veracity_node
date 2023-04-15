from typing import Optional, List, Union, Dict

from haystack import BaseComponent
from haystack.nodes import PromptNode, PromptModel


class VeracityNode(BaseComponent):
    outgoing_edges = 1

    def __init__(
            self,
            model_name_or_path: Union[str, PromptModel],
            api_key: Optional[str] = None,
            model_kwargs: Optional[Dict] = None,
    ):
        """
        Creates a VeracityNode

        :param model_name_or_path: The name of the model to use or an instance of the PromptModel.
        :param api_key: The API key to use for the model.
        :param model_kwargs: Additional keyword arguments passed when loading the model specified in `model_name_or_path`.
        """
        super().__init__()
        self.api_key = api_key
        self.model_name_or_path = model_name_or_path
        self.model_kwargs = model_kwargs

    def run(self, query: Union[List[str], str], results: Union[List[str], str], **kwargs):
        """
        Runs the fact checking. It asks the given model whether the given query is answered by the given context which
        are passed in results.

        :param query: The query to answer
        :param results: The given context
        :param kwargs: Additional keyword arguments, which will be passed as-is
        :return: The given query and context if the answer was correct, otherwise the given query a message that the answer
        was incorrect.
        """
        if query is None:
            return ValueError("query is None")
        if results is None:
            return ValueError("results is None")
        prompt = f"""
            You are given this context:
            {query}
            This is the question:
            {results}
            Did the provided answer correctly answer the question? 
            You should only reply with "True" or "False" and nothing else.
            """
        prompt_node = PromptNode(
            model_name_or_path=self.model_name_or_path, api_key=self.api_key, model_kwargs=self.model_kwargs
        )
        result_from_prompt_node = prompt_node(prompt)

        output_dict = {"query": query, "results": results}
        output_dict.update(kwargs)

        if "true" in result_from_prompt_node[0].lower():
            return output_dict, "output_1"
        else:
            output_dict["results"] = "The question was not answered correctly"
            return output_dict, "output_1"

    def run_batch(self, **kwargs):
        return NotImplementedError("Batch mode not implemented for FactcheckNode")
