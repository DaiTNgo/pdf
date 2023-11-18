from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import SentenceSplitter

documents = SimpleDirectoryReader("./pdf").load_data()
text_splitter = SentenceSplitter(
    separator=" ",
    chunk_size=1024,
    chunk_overlap=20,
    paragraph_separator="\n\n\n",
    secondary_chunking_regex="[^,.;。]+[,.;。]?",
)

node_parser = SimpleNodeParser.from_defaults(text_splitter=text_splitter)

# ---------
from llama_index.embeddings import HuggingFaceEmbedding

# embed_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embed_model_name = "BAAI/bge-base-en-v1.5"

embed_model = HuggingFaceEmbedding(model_name=embed_model_name, embed_batch_size=32)

# ---------
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
import chromadb

chroma_client = chromadb.PersistentClient(path="./dbb")
# chroma_client.delete_collection(name="qa-pdf")
chroma_collection = chroma_client.get_or_create_collection(name="qa-pdf")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ---------

from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM

system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

llm_model = "HuggingFaceH4/zephyr-7b-beta"
# llm_model = "MBZUAI/LaMini-T5-738M"

llm = HuggingFaceLLM(
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=llm_model,
    model_name=llm_model,
    device_map="auto",
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": False},
    stopping_ids=[50278, 50279, 50277, 1, 0],
    tokenizer_kwargs={"max_length": 4096},
)

# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# from transformers import pipeline
# from llama_index.llms import HuggingFaceInferenceAPI
# pipe = pipeline("text-generation", model=llm_model, torch_dtype=torch.bfloat16, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(llm_model)
# model = AutoModelForCausalLM.from_pretrained(llm_model)

# ---------
from llama_index import ServiceContext

service_context = ServiceContext.from_defaults(
    embed_model=embed_model,
    node_parser=node_parser,
    llm=llm,
)

# ---------
from llama_index import VectorStoreIndex

index = VectorStoreIndex.from_documents(
    documents=documents,
    service_context=service_context,
    storage_context=storage_context,
)
