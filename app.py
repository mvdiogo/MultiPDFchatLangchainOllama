import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredHTMLLoader,
    WebBaseLoader,
)
import os
import tempfile
from langchain_community.vectorstores.pgvector import PGVector
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Assign variables from environment variables
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
dbname = os.getenv('DB_NAME')
model = os.getenv('MODEL')

CONNECTION_STRING = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}'  # ?sslmode=require"


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ['Hello! Ask me anything about 🤗']

    if 'past' not in st.session_state:
        st.session_state['past'] = ['Hey! 👋']


def conversation_chat(query, chain, history):
    result = chain({'question': query, 'chat_history': history})
    history.append((query, result['answer']))
    return result['answer']


def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input(
                'Question:', placeholder='Ask about your PDF', key='input'
            )
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(
                    user_input, chain, st.session_state['history']
                )

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(
                    st.session_state['past'][i],
                    is_user=True,
                    key=str(i) + '_user',
                    avatar_style='thumbs',
                )
                message(
                    st.session_state['generated'][i],
                    key=str(i),
                    avatar_style='fun-emoji',
                )


def create_conversational_chain(vector_store):
    llm = Ollama(
        model=model,
        top_p=1,
        temperature=0.1,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={'k': 4}),
        memory=memory,
    )
    return chain


def main():
    # Initialize session state
    initialize_session_state()
    st.title('Local Chat using Ollama, Llama2 and pgvector')
    # Initialize Streamlit
    st.sidebar.title('Document Processing')
    uploaded_files = st.sidebar.file_uploader(
        'Upload files', accept_multiple_files=True
    )

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == '.pdf':
                loader = PyPDFLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000, chunk_overlap=200
        )
        text_chunks = text_splitter.split_documents(text)

        # Create embeddings
        embeddings = OllamaEmbeddings(model='nomic-embed-text')

        # Create vector store
        print('vector_store')
        vector_store = PGVector.from_documents(
            embedding=embeddings,
            documents=text_chunks,
            collection_name=file.name,
            connection_string=CONNECTION_STRING,
            pre_delete_collection=True,
        )
        # store = PGVector(
        #    collection_name=COLLECTION_NAME,
        #    connection_string=CONNECTION_STRING,
        #    embedding_function=embeddings,
        # )
        # store.add_documents([Document(page_content="foo")])

        # Create the chain object
        chain = create_conversational_chain(vector_store)

        display_chat_history(chain)


if __name__ == '__main__':
    main()
