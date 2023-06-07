import os
import json
import tiktoken
import openai
import numpy as np
from dotenv import load_dotenv
from openai.embeddings_utils import cosine_similarity
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Load environment variables
load_dotenv()

# Configure Azure OpenAI Service API
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define embedding model and encoding
EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_ENCODING = 'cl100k_base'
EMBEDDING_CHUNK_SIZE = 8000
COMPLETION_MODEL = 'text-davinci-003'

# initialize tiktoken for encoding text
encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)

# list all files in the samples directory
samples_dir = os.path.join(os.getcwd(), "../data/")
sample_files = os.listdir(samples_dir)

# read each file and remove and newlines (better for embeddings later)
documents = []
for file in sample_files:
    with open(os.path.join(samples_dir, file), "r") as f:
        content = f.read()
        content = content.replace("\n", " ")
        content = content.replace("  ", " ")
        documents.append(content)

# print some stats about the documents
print(f"Loaded {len(documents)} documents")
for doc in documents:
    num_tokens = len(encoding.encode(doc))
    print(f"Content: {doc[:80]}... \n---> Tokens: {num_tokens}\n")

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text):
    return openai.Embedding.create(input=text, engine=EMBEDDING_MODEL)["data"][0]["embedding"]

# Create embeddings for all docs
embeddings = [get_embedding(doc) for doc in documents]

# print some stats about the embeddings
for e in embeddings:
    print(len(e))

# create embedding for question
question = "what is azure openai service?"
qe = get_embedding(question)

# calculate cosine similarity between question and each document
similaries = [cosine_similarity(qe, e) for e in embeddings]

# Get the matching document, in this case we just use argmax of similarities
max_i = np.argmax(similaries)

# print some stats about the similarities
for i, s in enumerate(similaries):
    print(f"Similarity to {sample_files[i]} is {s}")
print(f"Matching document is {sample_files[max_i]}")

# Generate a prompt that we use for completion, in this case we put the matched document and the question in the prompt
prompt = f"""
Content:
{documents[max_i]}
Please answer the question below using only the content from above. If you don't know the answer or can't find it, say "I couldn't find the answer".
Question: {question}
Answer:"""

# get response from completion model
response = openai.Completion.create(
    engine=COMPLETION_MODEL,
    prompt=prompt,
    temperature=0.7,
    max_tokens=500,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None
)
answer = response['choices'][0]['text']

# print the question and answer
print(f"Question was: {question}\nRetrieved answer was: {answer}")
