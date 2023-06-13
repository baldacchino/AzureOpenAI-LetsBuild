import os
import tiktoken
import openai
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai.embeddings_utils import cosine_similarity


# Load environment variables
load_dotenv()

openai.api_type = "azure"
openai.api_base = os.environ.get("OPENAI_API_BASE")
openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.api_version = "2022-12-01"

# Define embedding model and encoding
EMBEDDING_MODEL = 'text-embedding-ada-002'
COMPLETION_MODEL = 'text-davinci-003'
encoding = tiktoken.get_encoding('cl100k_base')

print("--- Start : Testing Completition ---")
response = openai.Completion.create(engine="text-davinci-003",
                                    prompt="Knock knock.",
                                    temperature=0)
print(response.choices[0].text)
print("--- Complete : Testing Completion ---")
print("")
print("--- Start : Testing Completion Streaming ---")
import sys
for resp in openai.Completion.create(engine='text-davinci-003', prompt='Give me 10 taglines for an ice cream shop', max_tokens=512, stream=True):
    sys.stdout.write(resp.choices[0].text)
    sys.stdout.flush()
print("--- Complete : Testing Completion Streaming ---")
print("")
print("--- Start : Testing Embeddings ---")
e = openai.Embedding.create(input="Hello World!", engine=EMBEDDING_MODEL)["data"][0]["embedding"]
print(e)
print("--- Start : Testing Tokeniser ---")
tokens = encoding.encode("Hello world!")
print(tokens)
print(len(tokens))
print("--- Tokeniser : Testing Tokeniser ---")