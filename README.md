# MYOB Open AI WorkShop - Lets Get Building🤖 
So you know the basics around OpenAI but where to from here? 🙄

-------------------


In this session we will start to build, we will leverage the Open AI Python SDK to build some simple python based applications. These applications will cover multiple use cases. 

* 🔎Private Chat Bot : StreamLit + Prompt chaining you to build your own private Azure OpenAI based chatbot
  * Create a custom bot with a GPT like experience using your Azure Open AI end-point. This script will introduce you to Steamlit, a lightweight approach you can use to MVP from idea to concept and along with the basics in the Azure Open AI Python SDK
* 🤔Q&A Bot : Semantic search on your custom data.
  * What approaches can you use to bring your own data in to OpenAI. This script will demonstrate how you can set loose and provide context to Azure Open AI using your own data
* 🎥Classification and Sentiment Analysis : Embeddings, Keras, Clusters and LLM's
  * In this exampple we will leverage Keras to create a clusters from embeddings before providing sentiment analysis over our cluster data
  * Top 10K movies, reduced to a dataset of comedy and englishEmbedding all the keywords, taglines and titles
  * Sentiment analysis on the clusters using keras. 
  * Example given some keywords, provide supporting evidence. 


-------------------

## Preparation 🧑🏼‍💻 

> *This is only required if you are building alongside me. If you are only attending the presentation, you can skip this section.

During this session I will skim through content but explain examples in through the lens of possible MYOB uses cases.
Considerable amount of time in the IDE (Visual Studio Code) and I would welcome you all to build along with me.  You don't need to but you can leverage this boiler plater code for your hackathon and beyond. 

Prior to joining this session you will need to following to play along with my demos 

* ✅ Python 3
* ✅ Environment file with Azure Open AI credientals (.env file)
* ✅ Pip
* ✅ The following modules installed (pip3 install -r requirements.txt) - Found in the scripts folder
* ✅ Azure Open AI EndPoint
* ✅ Azure Open AI API-Key
* ✅ Following models deployed (gpt-35-turbo | code-davinci-002 | text-davinci-003 | text-embedding-ada-002)
* ✅ Recommendation you use WSL2 (Windows SubSystem For Linux) or a Linux based OS (this has not been tested under Windows)

-------------------

## Solutions

* :bulb: Script - [GPT3-5 Turbo Chat Bot](scripts/chatgpt_app.py)
* :bulb: Script - [Q&A with embeddings](scripts/qna_with_embeddings.py)
* :bulb: Script - [Unsupervised movie classification and recommendations](scripts/movie_classification.py)
