from flask import Flask

app = Flask(__name__)



# pip install --upgrade --quiet  langchain langchain-openai faiss-cpu tiktoken pypdf langchain_ai21

import os
os.environ["AI21_API_KEY"] = "dossBFzKy0rlsEtqWbNJvlBQHzKWhIhQ"
os.environ['OPENAI_API_KEY'] = "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"

from operator import itemgetter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# from PDF get TEXT and IMAGES
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("biology.pdf", extract_images=False) # set to false to do not get image's text
documents = loader.load() # or load_and_split()
documents[4].page_content


# documents to documents    (based on semantic)

#from langchain_ai21 import AI21SemanticTextSplitter
#from langchain_core.documents import Document
#semantic_text_splitter = AI21SemanticTextSplitter()
#documents = semantic_text_splitter.split_documents(documents) # it takes list of documents and return also list of documents
#documents[5].page_content


documents_texts = []
for i in documents:
  documents_texts.append(i.page_content)
print(documents_texts[5])

vectorstore = FAISS.from_texts(documents_texts, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

template = """You are an educational assistant in Ekson and you should define your self as that,
you do not say things like [Based on the context provided],
Answer the question based only on the following context:
{context}

if any one asked you to answer a question that is very far away of information in the context you should say [Sorry, I have no answer.],

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

my_reply = chain.invoke("what is BIO ?")





@app.route('/<singleMassage>')
def index(singleMassage):
    my_reply = chain.invoke(singleMassage)
    return my_reply

app.run(host='0.0.0.0', port=81)