import os
import chromadb
import openai
import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

os.environ['OPENAI_API_KEY'] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]


def get_pdf_text(pdf_docs):
    text = ""
    print(pdf_docs)
    for pdf in pdf_docs:
        print(pdf)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_embed_model():
    embed_model_name = "BAAI/bge-base-en-v1.5"
    return SentenceTransformerEmbeddings(model_name=embed_model_name, cache_folder='s_bert')

def save_db(text_chunks):
    embeddings = get_embed_model()
    vectorstore = Chroma.from_documents(text_chunks, embeddings,persist_directory="./db_lc")
    return vectorstore

def get_db():
    return Chroma(persist_directory="./db_lc", embedding_function=get_embed_model())

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
    )

    return conversation_chain


def process_answer_lc(user_question):
    response = get_conversation_chain(get_db())


def process_answer_lc():
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    # get pdf text
    raw_text = get_pdf_text(pdf_docs)

    # get the text chunks
    text_chunks = get_text_chunks(raw_text)

    # create vector store
    vectorstore = save_db(text_chunks)

    # create conversation chain
    st.session_state.conversation = get_conversation_chain(
        vectorstore)
