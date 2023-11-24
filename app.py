# Import necessary libraries
import openai
import streamlit as st
from bs4 import BeautifulSoup
import requests
import pdfkit
import time
from langchain.vectorstores import Chroma, Pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import os
from langchain.chains.question_answering import load_qa_chain
import PyPDF2
import pinecone
from PyPDF2 import PdfReader
from openai import OpenAI
from langchain.memory import MongoDBChatMessageHistory
from langchain.prompts import PromptTemplate
from streamlit_chat import message
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


promptText=""""

You are programmed to function as a chatbot. Your task is to answer to user questions based on the information provided in a conversation log and a specified knowledge base.You must not rely on external knowledge or infer answers beyond these documents. When formulating responses, adhere to these guidelines:

- Base your responses exclusively on the content of the provided documents. Do not incorporate external information or your pre-existing knowledge base.

- When facing a direct question, respond accordingly by matching the format of the question. If the question pertains to specific entities such as names, places, or dates, provide a straightforward answer by naming the entity. If the question seeks a description, ensure your response is sufficiently descriptive to facilitate clear understanding.

- Always prioritise the user's direct query. Focus on the specific question asked, rather than the broader context of the conversation.

- Only consider parts of the conversation log that are directly relevant to the user's immediate question.

- Remove any punctuation and non-essential words from the question to grasp its core intent.

- If subquestions arise related to your previous answers, respond based on the information available in the documents. Rely on your general abilities for basic calculations or logical deductions as needed.

- If the information required to answer a query is not available in the provided documents, clearly communicate that the answer is beyond the scope of the available data.

- Refrain from making assumptions or inferences when the documents do not explicitly provide the information needed to answer a question.

- Indicate the level of confidence in your responses based on the clarity and completeness of the information found in the documents.

Your role is to be a reliable intermediary between the user and the information contained within specific documents, using your general processing abilities to enhance understanding and answer precision.

  USER QUESTION: {userPrompt}

  CONVERSATION LOG: {conversationHistory}

  KNOWLEDGE BASE:{knowledgeBase}
  
  Final answer:"""




text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
client = openai


if "start_chat" not in st.session_state:
    st.session_state.start_chat = False

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if 'selected_items' not in st.session_state:
    st.session_state['selected_items'] = []

message_history=None
# Set up the Streamlit page with a title and icon
st.set_page_config(page_title="ChatGPT-like Chat App", page_icon=":speech_balloon:")

def create_pinecone_index(index_name):
  print(pinecone.list_indexes())
  if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=1536
    )
    while not pinecone.describe_index(index_name).status["ready"]:
        time.sleep(1)
  return index_name



def get_embedding(text_chunk):
    response = client.embeddings.create(
        input=text_chunk,
        model="text-embedding-ada-002")
    return response.data[0].embedding


def run_semantic_search(query):
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    docs_list=[]
    docs = docsearch.similarity_search(query)
    for doc in docs:
      docs_list.append(doc)
    return docs_list

def get_prompt(prompt_template,question,converstion,knowledge):
    final_promt=prompt_template.format(userPrompt=question, conversationHistory=converstion, knowledgeBase=knowledge)
    return final_promt
 
def get_response_from_openai(prompt_in):
    try:
        client = OpenAI()
        response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
            "role": "user",
            "content": prompt_in
            }
        ],
        temperature=0,
        max_tokens=3000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        # return response.choices[0].text
        return response.choices[0].message.content
    except:
        return "issue With Opean AI"

def get_file_name_pinecone():
    index = pinecone.Index(index_name)
    sample_vector=[0.3 for i in range(1536)]
    vec_data=index.query(
            vector=sample_vector,
            top_k=10000,
            include_metadata=True,
    )
    unique_filenames = {match['metadata']['filename'] for match in vec_data['matches']}
    unique_filenames = list(unique_filenames)
    return unique_filenames


def upload_vector_to_pinecone(data,name_file):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
  texts = text_splitter.split_documents(data)
  embeddings = OpenAIEmbeddings()
  metadatas = []
  for text in texts:
        metadatas.append({
            "filename": name_file
        })
  Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name, metadatas=metadatas)

# Create a sidebar for MongoDB key configuration 
st.sidebar.header("MongoDB Configuration")
connection_string = st.sidebar.text_input("Enter your MongoDB Connection String", type="password")
if connection_string:
    message_history = MongoDBChatMessageHistory(
    connection_string=connection_string, session_id="test-session"
)

# Create a sidebar for OpenAI key configuration 
st.sidebar.header("Opean ai Configuration")
api_key_openai = st.sidebar.text_input("Enter your OpenAI API key", type="password")
if api_key_openai:
    openai.api_key = api_key_openai
    os.environ['OPENAI_API_KEY'] = api_key_openai

# Create a sidebar for Pinecone key configuration 
st.sidebar.header("Pinecone Configuration")
PINECONE_API_KEY = st.sidebar.text_input("Enter your Pinecone API key", type="password")
PINECONE_API_ENV = st.sidebar.text_input("Enter your Pinecone Environment", type="password")
unique_filenames_list=[]
if PINECONE_API_KEY and PINECONE_API_ENV:
    pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV)
    index_name="bottest"
    create_pinecone_index(index_name)
    unique_filenames_list=get_file_name_pinecone()


# Sidebar option for users to upload their own files
uploaded_files = st.sidebar.file_uploader("Upload a files to pinecone", key="file_uploader",accept_multiple_files=True, type="pdf")
file_name_list_new=[]
if st.sidebar.button("Upload File"):
    if uploaded_files:
       for file in uploaded_files:
            bytes_data =file.read()
            _ , file_extension = os.path.splitext(file.name)
            with open(file.name,"wb") as f:
                f.write(bytes_data)
            loader = PyPDFLoader(file.name)
            data=loader.load()
            file_name=file.name
            file_name_list_new.append(file_name)
            upload_vector_to_pinecone(data,file_name)
           

# get filenames
unique_filenames_list_final=list(dict.fromkeys(unique_filenames_list+file_name_list_new))

def update_selected_items():
    st.session_state['selected_items'] = [
        item for item in unique_filenames_list_final if st.session_state[item]
    ]

# Create a checkbox for filenames
st.sidebar.header("Existing Files In Pinecone")
for item in unique_filenames_list_final:
    st.sidebar.checkbox(item, key=item, on_change=update_selected_items)


st.title("G37- Converse With Your Documents")
st.write("Chatbot for Querying Information from Your Documents")


# Display chat messages
if st.sidebar.button("Start Chat"):
    if unique_filenames_list:

            st.session_state.start_chat = True
            thread = client.beta.threads.create()
            st.session_state.thread_id = thread.id
            st.write("thread id: ", thread.id)        

    else:
        st.sidebar.warning("Please upload at least one file to start the chat.")

   
else:
    # Prompt to start the chat
    st.write("Please enter your API keys and upload the files. Then, select the files you wish to inquire about from Existing Files and click 'Start Chat' to initiate the conversation.")
if message_history:
    chat_history = message_history.messages
    st.session_state.messages =[{"role":"user","content":msg.content} if isinstance(msg, HumanMessage) else {"role":"ai","content":msg.content} for msg in chat_history]

# User-provided prompt
if st.session_state.start_chat:
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages in the chat
    for message in st.session_state.messages:
        print(message)
        if message["role"] == "user":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if chat_input := st.chat_input("Ask..."):
        message_history.add_user_message(chat_input)
        chat_history = message_history.messages
        combined_chat_history = "\n".join([f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" for msg in chat_history])
        embeddings = OpenAIEmbeddings()
        docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    
        retrieved_docs=[]
        for file in st.session_state['selected_items']:  
            docs = docsearch.similarity_search(chat_input, filter={"filename":file})
            for doc in docs:
                retrieved_docs.append(doc)

        if retrieved_docs:
          combined_docs = "\n\n".join([doc.page_content for doc in retrieved_docs])

        else:
          combined_docs="No relevant text available in the document"  
        
        prompt_for_bot=get_prompt(promptText,chat_input,combined_chat_history,combined_docs)
        ai_response=get_response_from_openai(prompt_for_bot)

        message_history.add_ai_message(ai_response)
        with st.chat_message("user"):
            st.markdown(chat_input)

        with st.chat_message("ai"):
            st.markdown(ai_response)


        st.session_state.messages.append({"role": "user", "content": chat_input})
  
        












