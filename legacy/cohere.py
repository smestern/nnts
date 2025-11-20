# pip install transformers
#set the environment variable 
import os
import time
os.environ["TRANSFORMERS_CACHE"] = "I:/hf/"
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True)




model_id = "CohereForAI/c4ai-command-r-v01"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision='102972a', trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(model_id, revision='102972a', trust_remote_code=False, quantization_config=bnb_config)

# define conversation input:
conversation = [
    {"role": "user", "content": "Whats the biggest penguin in the world?"}
]
# Define tools available for the model to use:
tools = [
  {
    "name": "internet_search",
    "description": "Returns a list of relevant document snippets for a textual query retrieved from the internet",
    "parameter_definitions": {
      "query": {
        "description": "Query to search the internet with",
        "type": 'str',
        "required": True
      }
    }
  },
  {
    'name': "directly_answer",
    "description": "Calls a standard (un-augmented) AI chatbot to generate a response given the conversation history",
    'parameter_definitions': {}
  }
]
call_time = time.time()

# render the tool use prompt as a string:
tool_use_prompt = tokenizer.apply_tool_use_template(
    conversation,
    tools=tools,
    tokenize=False,
    add_generation_prompt=True,
)

prompt = tokenizer.encode(tool_use_prompt, return_tensors='pt')
gen_tokens = model.generate(
    prompt, 
    max_new_tokens=100, 
    do_sample=True, 
    temperature=0.3,
    )

gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)
print(f"Generation took {time.time() - call_time:.2f} seconds")