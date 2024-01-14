from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema.document import Document

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def save_to_disk(docs):
    Chroma.from_documents(docs, embedding_function, persist_directory="./db_lc")


def load_from_disk():
    return Chroma(persist_directory="./db_lc", embedding_function=embedding_function)


import numpy as np


def get_documents(pdf_docs):
    documents = []

    for pdf in pdf_docs:
        document = get_pdf_text(pdf)
        # document = Document(
        #     page_content=get_pdf_text(pdf),
        #     metadata={"source": pdf.filename},
        # )
        documents.append(document)

    return np.array(documents).flatten()
    # return documents.flat()


def get_pdf_text(pdf):
    text = []

    pdf_reader = PdfReader(pdf.file)
    for index, page in enumerate(pdf_reader.pages):
        text.append(
            Document(
                page_content=page.extract_text(),
                metadata={
                    "source": pdf.filename,
                    "page": index
                }
            )
        )
        # text += page.extract_text()
    return text


def get_text_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)

    text_chunks = text_splitter.split_documents(data)
    # text_splitter = CharacterTextSplitter(
    #     separator="\n",
    #     chunk_size=1000,
    #     chunk_overlap=200,
    #     length_function=len
    # )
    # chunks = text_splitter.split_text(text)
    return text_chunks


def process_lc(files):
    documents = get_documents(files)
    docs = get_text_chunks(documents)
    save_to_disk(docs)


def process_answer_lc(instruction):
    vector_store = load_from_disk()

    llm = OpenAI(streaming=True, temperature=0.75, top_p=1, verbose=True)

    # Initialize RetrievalQA
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2})
    )

    # Run the query
    result = qa.run(instruction)

    return result
