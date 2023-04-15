# Haystack Veracity Node

> veracity: conformity to facts; accuracy

This Node checks whether the given input is correctly answered by the given context (as judged by the given LLM). One example usage is together with [Haystack Memory](https://github.com/rolandtannous/haystack-memory): After the memory is retrieved, the given model checks whether the output is satisfying the question. 

**Important**: 
The Node expects the context to be passed into `results`. If the previous node in the pipeline is putting the text somewhere else, use a [Shaper](https://docs.haystack.deepset.ai/docs/shaper) to `rename` the argument to `results`. 


## Example Usage with Haystack Memory
```py
from haystack_veracity_node.node import VeracityNode
from haystack_memory.memory import RedisMemoryRecallNode, memory_template
from haystack import Pipeline
from haystack.agents import Agent, Tool
from haystack.nodes import PromptNode

# Create VeracityNode
veracity_node = VeracityNode(model_name_or_path="gpt-3.5-turbo", api_key="YOUR_KEY")

# Create Memory
redis_memory_node = RedisMemoryRecallNode(memory_id="agent_memory",
                                          host="localhost",
                                          port=6379,
                                          db=0)

# Add them together in a pipeline
memory_pipeline = Pipeline()
memory_pipeline.add_node(component=redis_memory_node, name="MemoryTool", inputs=["Query"])
memory_pipeline.add_node(component=veracity_node, name="VeracityNode", inputs=["MemoryTool"])

# Create an agent and add the pipeline as a tool
prompt_node = PromptNode(model_name_or_path="text-davinci-003", api_key=openai_api_key, max_length=512,
                         stop_words=["Observation:"])
memory_agent = Agent(prompt_node=prompt_node, prompt_template=memory_template)
memory_tool = Tool(name="Memory",
                   pipeline_or_node=memory_pipeline,
                   description="Your memory. Always access this tool first to remember what you have learned.")

memory_agent.add_tool(memory_tool)
```