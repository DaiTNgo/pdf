import os

import chromadb
import openai
from PyPDF2 import PdfReader
from llama_index import ServiceContext, set_global_service_context, VectorStoreIndex, Document
from llama_index import SimpleDirectoryReader
from llama_index.chat_engine.types import ChatMode
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.llms import OpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts import PromptTemplate
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore
# 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
# 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
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

# save_index(SimpleDirectoryReader("./benchmarks_data_3").load_data())

eval_questions = [
    "What is machine learning?",
    "what is natural language processing?",
    "what is deep learning?",
    "How businesses are using machine learning?",
    "What is Retrieval Augmented Generation?",
    "Who was the first person to come up with the idea for RAG?"
]

eval_answers = [
  "Machine learning is a field of study that focuses on developing algorithms and models that enable computers to learn and make predictions or decisions without being explicitly programmed. It involves the use of statistical techniques to enable computers to learn from and analyze large amounts of data, identify patterns, and make predictions or decisions based on that data. Machine learning is often used in various applications such as image recognition, natural language processing, and recommendation systems.",
  "Natural language processing is a field of machine learning where machines learn to understand and interpret human language as it is spoken and written. Instead of relying on traditional programming methods that use data and numbers, natural language processing allows machines to recognize, understand, and respond to language. It also enables machines to generate new text and translate between different languages. Natural language processing is the technology behind familiar applications like chatbots and digital assistants such as Siri or Alexa.",
  'Deep learning refers to a type of machine learning that involves the use of neural networks with multiple layers. These neural networks are comprised of interconnected nodes, or artificial neurons, that have associated weights and thresholds. If the output of a node exceeds the specified threshold, it is activated and passes data to the next layer of the network. Deep learning algorithms typically have more than three layers, including an input and output layer, which is why they are considered "deep." Deep learning has been instrumental in advancing areas such as computer vision, natural language processing, and speech recognition. For a more detailed understanding of the differences between AI, machine learning, deep learning, and neural networks, you can refer to the blog post "AI vs. Machine Learning vs. Deep Learning vs. Neural Networks: What\'s the Difference?"',
  "Businesses are using machine learning in various ways. For example, they are using neural networks for natural language translation, image recognition, speech recognition, and image creation. Linear regression is being used to predict numerical values, such as house prices based on historical data. Logistic regression is being used for applications like classifying spam and quality control on a production line. Clustering algorithms are helping businesses identify patterns in data that humans may have overlooked. Decision trees are being used for predicting numerical values and classifying data into categories. Random forests are being used to predict values or categories by combining the results from multiple decision trees.",
  "Retrieval-Augmented Generation (RAG) is a technique used in generative artificial intelligence (AI) that involves creating text responses based on large language models (LLMs). The AI is trained on a massive amount of data points, which allows it to generate text that is often easy to read and provides detailed responses. RAG works by using information from the training data to generate the response. However, one limitation of RAG is that the information used to generate the response is limited to the information used to train the AI, which can be weeks, months, or even years out of date. In the context of a corporate AI chatbot, this means that the AI may not have specific information about the organization's products or services, which can lead to incorrect responses.",
  "Sebastian Riedel was the first person to come up with the idea for RAG.",
]

def get_eval_questions(q):
    res = process_answer(q)
    text = ''
    for token in res.response_gen:
        text += token
    return text

import nest_asyncio
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llama_index import evaluate

documents = SimpleDirectoryReader("benchmarks_data_3").load_data()
vector_index = VectorStoreIndex.from_documents(
    documents, service_context=ServiceContext.from_defaults(chunk_size=512)
)
query_engine = vector_index.as_query_engine()
nest_asyncio.apply()
eval_answers = [[a] for a in eval_answers]

metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]
# harmfulness,
# result = evaluate(query_engine, metrics, eval_questions, eval_answers)

import time
result = []
# time.sleep(60)
# result = evaluate(query_engine, metrics, eval_questions, eval_answers)
for i in range(len(eval_answers)):
    rs = evaluate(query_engine, metrics, eval_questions[i], eval_answers[i])
    print(rs)
    result.append(rs)
    time.sleep(60)

print(result)