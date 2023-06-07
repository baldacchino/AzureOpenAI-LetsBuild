# MYOB Open AI WorkShop - Lets Get Building🤖 - So you know the basics around OpenAI but where to from here? 

> In this session we will start to build, we will leverage the Open AI Python SDK to build some simple python based applications. These applications will cover multiple use cases. 
1. Private Chat Bot : This will illustrate how using StreamLit with prompt chaining you cna build your own private LLM based chatbot
2. Q&A Bot that with semantic search on a set of sample data : Want to set the LLM's loose of your data, this pattern will show you how to do this.
3. Classification and Sentiment Analysis : Building on embeddings, we will leverage Keras to create a vector DB's and 
Top 10K movies, reduced to a dataset of comedy and englishEmbedding all the keywords, taglines and titles
Sentiment analysis on the clusters using keras. 
Example given some keywords, provide supporting evidence. 
This wil tie in to the example Darren (Or Adrian gave) on understanding churn rates at MY
In this technical workshop, you will get a comprehensive introduction to Azure OpenAI Service and Azure OpenAI Studio. You will learn how to create and refine prompts for various scenarios using hands-on exercises. You will also discover how to leverage Azure OpenAI Service to access and analyze your company data. Moreover, you will explore existing solution accelerators and best practices for prototyping and deploying use cases end-to-end. The workshop will end with a Q&A session and a wrap-up.*

## Workshop agenda


<sup>
📣 Presentation, 🧑🏼‍💻 Hands-on lab
</sup>

-------------------

## Preparation

> *This is only required for the hands-on lab. If you are only attending the presentation, you can skip this section.*

### Azure OpenAI Service subscription and deployments

During this session I will skim through content but explain these examples in to possible MYOB uses cases.
I will spend a considerable amount of time in the IDE (Visual Studio Code) and I would welcome you all to build along with me.  You don't need to but I want to set you with some basic boiler plate code for your event.
Prior to joining this session you will need to following to play along with my demos (source code will be shared)
✅ Python 3
✅ Pip
✅ The following modules installed (pip install azure-identity streamlit openai python-dotenv numpy pandas matplotlib plotly scipy scikit-learn tenacity tiktoken llama-index langchain faiss)
✅ Azure Open AI EndPoint
✅ Azure Open AI API-Key
✅ Following models deployed (gpt-35-turbo | code-davinci-002 | text-davinci-003 | text-embedding-ada-002)
✅ Recommendation you use WSL2 (Windows SubSystem For Linux) or a Linux based OS (this has not been tested under Windows)

-------------------

## Content of the repository

* :bulb: [Guideline for writing better prompts](lectures/prompt_writing_help.md)

## Exercises

* :muscle: [Simple prompt writing exercises](exercises/exercises.md)
* :muscle: [Quickstart](exercises/quickstart.ipynb) - just to make sure everything works!
* :muscle: [Preprocessing](exercises/preprocessing.ipynb) - principles of preprocessing and token handling!
* :muscle: [Q&A with embeddings](exercises/qna_with_embeddings_exercise.ipynb)
* :muscle: [Unsupervised movie classification and recommendations](exercises/movie_classification_unsupervised_incl_recommendations_exercise.ipynb)
* :muscle: [Email Summarization and Answering App](exercises/email_app.md)

## Solutions

Do not cheat! :sweat_smile:

* :bulb: Solution - [Q&A with embeddings](exercises/solutions/qna_with_embeddings_solution.ipynb)
* :bulb: Solution - [Unsupervised movie classification and recommendations](exercises/solutions/movie_classification_unsupervised_incl_recommendations_solution.ipynb)
* :bulb: Solution - [Email Summarization and Answering App](exercises/solutions/email_app.py)

## Q&A Quick Start

If you want to quickly create a Q&A webapp using your own data, please follow the [quickstart guide notebook](qna-quickstart-template/qna-app-quickstart.ipynb).

If you want to use LangChain to build an interactive chat experience on your own data, follow the [quickstart chat on private data using LangChain](qna-chat-with-langchain/qna-chat-with-langchain.ipynb).

If you want to use LlamaIndex 🦙 (GPT Index), follow the [quickstart guide notebook with llama-index](qna-quickstart-with-gpt-index/qna-quickstart-with-llama-index.ipynb).
