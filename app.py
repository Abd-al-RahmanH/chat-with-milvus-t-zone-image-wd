import streamlit as st
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
import os
from dotenv import load_dotenv

# Initialize Environment
load_dotenv("config.env")

# Milvus Connection
milvus_host = os.getenv("MILVUS_HOST")
milvus_port = os.getenv("MILVUS_PORT")
cert_file_path = "cert.crt"  # Assuming `cert.crt` is in the same directory as app.py

connections.connect(
    alias="default",
    host=milvus_host,
    port=milvus_port,
    user="ibmlhadmin",
    password="password",
    secure=True,
    server_pem_path=cert_file_path,
    server_name="watsonxdata"
)
collection_name = "wiki_articles"
collection = Collection(collection_name)
collection.load()

# IBM Watsonx Credentials
creds = {
    "url": os.getenv("IBM_CLOUD_URL"),
    "apikey": os.getenv("API_KEY")
}
project_id = os.getenv("PROJECT_ID")

# Sentence Transformer Model
sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# LLM Model Setup
llm_params = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 500,
    GenParams.TEMPERATURE: 0,
}
llm_model = Model(
    model_id="ibm/granite-13b-chat-v2", params=llm_params, credentials=creds, project_id=project_id
)

# Streamlit App
st.title("Retrieval-Augmented Generation (RAG) Bot")
st.write("Ask a question, and the bot will retrieve relevant information from Milvus and provide a detailed response.")

# User Input
query = st.text_input("Enter your question:")

if query:
    # Vectorize Query
    query_vector = sentence_model.encode([query])

    # Search in Milvus
    search_params = {"metric_type": "L2", "params": {"nprobe": 5}}
    results = collection.search(
        data=query_vector, anns_field="vector", param=search_params, limit=5, output_fields=["article_text"]
    )
    
    # Prepare Context
    context = "\n\n".join([result.entity.get("article_text") for result in results[0]])
    
    # Generate Prompt
    prompt = f"{context}\n\nPlease answer the question using the above text. Question: {query}"
    
    # Get LLM Response
    response = llm_model.generate_text(prompt)
    
    # Display Results
    st.subheader("Answer")
    st.write(response)

    st.subheader("Supporting Passages")
    for idx, passage in enumerate(context.split("\n\n"), 1):
        st.write(f"Passage {idx}: {passage}")
