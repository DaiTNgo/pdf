# from llama_index.node_parser import SimpleNodeParser
# from llama_index.text_splitter import SentenceSplitter
#
# text_splitter = SentenceSplitter(
#     separator=" ",
#     chunk_size=1024,
#     chunk_overlap=20,
#     paragraph_separator="\n\n\n",
#     secondary_chunking_regex="[^,.;。]+[,.;。]?",
# )
#
# node_parser = SimpleNodeParser.from_defaults(text_splitter=text_splitter)
#
# # ---------
# from llama_index.embeddings import HuggingFaceEmbedding
# # embed_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
# embed_model_name = "BAAI/bge-base-en-v1.5"
#
# embed_model = HuggingFaceEmbedding(model_name=embed_model_name, embed_batch_size=32)
#
# # ---------
# from llama_index.vector_stores import ChromaVectorStore
# from llama_index.storage.storage_context import StorageContext
# import chromadb
#
# chroma_client = chromadb.PersistentClient(path="./dbb")
# chroma_collection = chroma_client.get_or_create_collection(name="qa-pdf")
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
#
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
#
# # ---------
#
# from llama_index.prompts import PromptTemplate
# from llama_index.llms import HuggingFaceLLM, HuggingFaceInferenceAPI
#
#
# system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
# - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
# - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
# - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
# - StableLM will refuse to participate in anything that could harm a human.
# """
#
# # This will wrap the default prompts that are internal to llama-index
# query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")
#
# llm_model = "HuggingFaceH4/zephyr-7b-beta"
# # llm_model = "MBZUAI/LaMini-T5-738M"
#
# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
# model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
#
# llm = HuggingFaceLLM(
#     system_prompt=system_prompt,
#     query_wrapper_prompt=query_wrapper_prompt,
#     tokenizer_name=llm_model,
#     model_name=llm_model,
#     device_map="auto",
#     context_window=4096,
#     max_new_tokens=256,
#     generate_kwargs={"temperature": 0.7, "do_sample": False},
#     stopping_ids=[50278, 50279, 50277, 1, 0],
#     tokenizer_kwargs={"max_length": 4096},
# )
#
# # ---------
# from llama_index import ServiceContext
#
# service_context = ServiceContext.from_defaults(
#     embed_model=embed_model,
#     node_parser=node_parser,
#     llm=llm,
# )
#
# # ---------
# from llama_index import VectorStoreIndex
#
# index = VectorStoreIndex.from_vector_store(
#     vector_store=vector_store,
#     service_context=service_context,
# )

# query_engine = index.as_query_engine()
# response = query_engine.query("Những tính chất của chất bán dẫn?")
# print(response)

# ------------------------

import base64
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)
# ----------------------
import os
def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size
# ------------------------
import streamlit as st


def main():
    st.markdown("<h1 style='text-align: center; color: blue;'>Q/A with PDF 📄</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Upload your PDF 👇</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["pdf"])

    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": get_file_size(uploaded_file)
        }
        filepath = uploaded_file.name
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("<h4 style color:black;'>File details</h4>", unsafe_allow_html=True)
            st.json(file_details)
            st.markdown("<h4 style color:black;'>File preview</h4>", unsafe_allow_html=True)
            pdf_view = displayPDF(filepath)

        with col2:
            with st.spinner('Embeddings are in process...'):
                ingested_data = ""#data_ingestion()
            st.success('Embeddings are created successfully!')
            st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)

            user_input = st.text_input("", key="input")



if __name__ == "__main__":
    main()