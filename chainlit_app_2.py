import os
import chainlit as cl
import chromadb
import openai
from chainlit.types import AskFileResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index import ServiceContext, set_global_service_context, VectorStoreIndex, Document
from llama_index.chat_engine.types import ChatMode
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.llms import OpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts import PromptTemplate
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore
from langchain_app import process_lc, process_answer_lc

os.environ['OPENAI_API_KEY'] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]

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

from PyPDF2 import PdfReader


def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf.file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_documents(pdf_docs):
    documents = []

    for pdf in pdf_docs:
        document = Document(
            text=get_pdf_text(pdf),
            metadata={
                "file_name": pdf.filename,
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
    service_context = ServiceContext.from_defaults(
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
    return VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=get_storage_context(),
        vector_store=get_vector_store(),
        service_context=get_service_context(),
        text_qa_template=text_qa_template,
        refine_template=refine_template,
        postprocessor=rerank(),
        show_progress=True,
    )


def process_answer(instruction):
    index = get_index()

    chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONDENSE_QUESTION,
        verbose=True,
    )
    streaming_response = chat_engine.stream_chat(instruction)
    return streaming_response


@cl.on_chat_start
async def on_chat_start():
    files = None
    #
    # # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!", accept=["application/pdf"]
        ).send()

    file = files[0]
    get_docsearch(file)

    cl.user_session.set(
        "message_history",
        [],
    )


class Message(cl.Message):
    type: str
    unique: str

    def __init__(self, content="", type='', unique='', **rest):
        self.type = type
        self.unique = unique
        super().__init__(content, **rest)

    def to_dict(self):
        dir = super().to_dict()
        dir['type'] = self.type
        dir['unique'] = self.unique
        return dir


import asyncio


async def llama(content, msg_llama):
    stream = process_answer(content)
    for token in stream.response_gen:
        # await asyncio.gather(msg_llama.stream_token(token), msg_chain.stream_token(token))
        await msg_llama.stream_token(token)


async def langchain(content, msg_chain):
    stream = process_answer_lc(content)
    await msg_chain.stream_token(stream)

    # for token in stream.response_gen:
    #     await msg_chain.stream_token(token)
    #
    # return stream


@cl.on_message
async def on_message(message: Message):
    msg_llama = cl.Message(content="")
    await llama(message.content, msg_llama)
    await msg_llama.send()


# TODO: server
from chainlit.server import app
from fastapi.responses import (
    JSONResponse,
)
from fastapi import UploadFile
import time


@app.post("/process")
async def process(files: list[UploadFile]):
    # for langchain
    # process_lc(files)
    # for langchain

    documents = get_documents(files)
    start_time = time.time()
    print("--- %s seconds ---" % start_time)
    save_index(documents)
    # embeddings_lc(files)
    print("--- %s seconds ---" % (time.time() - start_time))
    return JSONResponse({
        "status": 200,
    })


from langchain.document_loaders import PyPDFLoader

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


def process_file(file: AskFileResponse):
    import tempfile
    Loader = PyPDFLoader
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tempfile:
        with open(tempfile.name, "wb") as f:
            f.write(file.content)

        loader = Loader(tempfile.name)
        documents = loader.load()

        # document = Document(
        #     text=documents,
        #     metadata={
        #         "file_name": 'hihi',
        #     },
        #     excluded_llm_metadata_keys=["file_name"],
        #     metadata_seperator="::",
        #     metadata_template="{key}=>{value}",
        #     text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
        # )



        docs = text_splitter.split_documents(documents)

        documents = []
        for i, doc in enumerate(docs):
            # doc.metadata["source"] = f"source_{i}"
            doc = Document(
                text=doc.page_content,
                metadata=doc.metadata,
                excluded_llm_metadata_keys=["file_name"],
                metadata_seperator="::",
                metadata_template="{key}=>{value}",
                text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
            )
            documents.append(doc)



        return documents


def get_docsearch(file: AskFileResponse):
    docs = process_file(file)
    # print('docs', docs)
    save_index(docs)
