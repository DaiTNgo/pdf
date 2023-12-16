import os

import chromadb
import openai
import streamlit as st
from dotenv import load_dotenv
from llama_index import ServiceContext, set_global_service_context, SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.chat_engine.types import ChatMode
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.llms import OpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts import PromptTemplate
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore

from htmlTemplates import css, bot_template, user_template

os.environ['OPENAI_API_KEY'] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]

print('re-run')

text_qa_template_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using both the context information and also using your own knowledge, "
    "answer the question: {query_str}\n"
    "If the context isn't helpful, you can also answer the question on your own. You can choose answer the question by Vietnamese or English\n"
)
text_qa_template = PromptTemplate(text_qa_template_str)

# Refine template
refine_template_str = (
    "The original question is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Using both the new context and your own knowledge, update or repeat the existing answer.\n"
)
refine_template = PromptTemplate(refine_template_str)

from llama_index.readers.file.docs_reader import DocxReader, HWPReader, PDFReader

from PyPDF2 import PdfReader

def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_documents(pdf_docs):
    return SimpleDirectoryReader('./pdf').load_data()
    documents = []
    for pdf in pdf_docs:
        document = Document(
            text=get_pdf_text(pdf),
            metadata={
                "file_name": pdf.name,
            },
            excluded_llm_metadata_keys=["file_name"],
            metadata_seperator="::",
            metadata_template="{key}=>{value}",
            text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
        )
        documents.append(document)

    return documents


def get_node_parser():
    return SimpleNodeParser.from_defaults()


def get_embed_model():
    embed_model_name = "BAAI/bge-base-en-v1.5"
    return HuggingFaceEmbedding(model_name=embed_model_name, cache_folder='s_bert')


def get_llm_model():
    return OpenAI(model='gpt-3.5-turbo', max_tokens=512, temperature=0.1)


def get_service_context():
    service_context =  ServiceContext.from_defaults(
        embed_model=get_embed_model(),
        node_parser=get_node_parser(),
        llm=get_llm_model(),
    )
    set_global_service_context(service_context)
    return service_context


def get_vector_store():
    chroma_client = chromadb.PersistentClient(path="./dbb")
    chroma_collection = chroma_client.get_or_create_collection(name="qa-pdf")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store


def get_storage_context():
    storage_context = StorageContext.from_defaults(vector_store=get_vector_store())
    return storage_context


def rerank():
    postprocessor_model_name = "sentence-transformers/msmarco-distilbert-base-dot-prod-v3"
    postprocessor = SentenceTransformerRerank(
        model=postprocessor_model_name,
        top_n=3,
    )
    return postprocessor


def get_index():
    index = VectorStoreIndex.from_vector_store(
        storage_context=get_storage_context(),
        vector_store=get_vector_store()
    )
    return index


def save_index(documents):
    VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=get_storage_context(),
        vector_store=get_vector_store(),
        service_context=get_service_context(),
        text_qa_template=text_qa_template,
        refine_template=refine_template,
        postprocessor=rerank(),
    )


def process_answer(instruction):
    index = get_index()

    chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONDENSE_QUESTION,
        verbose=True,
    )

    print(instruction)

    streaming_response = chat_engine.stream_chat(instruction)

    response = ''
    for token in streaming_response.response_gen:
        response += token

    return response


# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        st.write(user_template.replace(
            "{{MSG}}", history['past'][i]), unsafe_allow_html=True)
        st.write(
            bot_template.replace("{{MSG}}", history['generated'][i]),
            unsafe_allow_html=True
        )


def handle_userinput(user_input):
    # Initialize session state for generated responses and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []

    # Search the database for a response based on user input and update session state
    if user_input:
        answer = process_answer(user_input)
        st.session_state["past"].append(user_input)
        response = answer
        st.session_state["generated"].append(response)

    # Display conversation history using Streamlit messages
    if st.session_state["generated"]:
        display_conversation(st.session_state)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type=["pdf"])

        if st.button("Process"):
            with st.spinner("Processing"):
                documents = get_documents(pdf_docs)

                save_index(documents)


if __name__ == '__main__':
    main()
