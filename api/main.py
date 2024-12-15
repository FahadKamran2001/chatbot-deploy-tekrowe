from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import logging
import os
import dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
import requests
import bs4
from bs4 import BeautifulSoup
import logging
import uvicorn
#logging.basicConfig(level=logging.DEBUG)
# Load environment variables (API Key)
dotenv.load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize Jinja2 template engine (create "templates" folder for HTML)
templates = Jinja2Templates(directory="templates")

# Define Pydantic models for request body validation
class QueryRequest(BaseModel):
    question: str

# Function to define OpenAI LLM
def define_llm(OPENAI_API_KEY):
    '''
      Provide An API Key for OpenAI to create an llm
    '''
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY
    )
    return llm

# Function to load website content into a vector store
def load_pdf_content_to_vectorstore(locations):
    '''
      Provide a list of urls for the content to be loaded
    '''
    loader = PyPDFLoader(locations)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=2048, chunk_overlap=256, add_start_index=True
    )
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return vectorstore

# Function to create the Conversational Chain
def create_chain(vectorstore, llm):
    '''
      Provide a vectorstore and an llm to create a chain
    '''
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectorstore.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
    )
    return qa

# Function to predict based on the message
def predict(qa, message):
    '''
    Provide a chain and a message to predict the response
    '''
    prompt = f'''
    Personality: You are a customer service agent on a company website tasked with answering customer questions regarding the company based on the information provided. Provide all answers as an agent.
                 Answer questions only regarding tekrowe, which is the company, and the information in the retrieval. No other question should be answered apart from the retrieved information.

    Question: {message}
    '''
    response = qa({"question": prompt})
    #assistant_message = response['answer']
    assistant_message = response.get('answer', 'Sorry, I am not able to answer that question.')
    return assistant_message

# Create base LLM and load content to vectorstore
llm = define_llm(os.getenv('OPENAI_API_KEY'))
print("Program finished successfully")
pdf = "Tekrowe_Dec2024.pdf"
vectorstore = load_pdf_content_to_vectorstore(pdf)
print("Program finished successfully")
qa = create_chain(vectorstore, llm)
print("Program finished successfully")
# FastAPI route to handle query requests
@app.post("/predict")
async def predict_query(question: str = Form(...)):
    try:
        # Get the prediction from the QA chain
        answer = predict(qa, question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# FastAPI route for HTML page with form
@app.get("/", response_class=HTMLResponse)
async def get_homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# FastAPI root route for basic info
@app.get("/info")
def read_root():
    return {"message": "Welcome to Tekrowe's customer service API!"}
