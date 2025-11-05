import streamlit as st
import os
import uuid
import pandas as pd
import numpy as np
import httpx
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_community.document_loaders import CSVLoader
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import json
import requests

# =========================
# Backend Logic (unchanged)
# =========================
load_dotenv()
client = httpx.Client(verify=False)

# Load KB documents
loader = CSVLoader(file_path="Y:/Hackathon/Chennai/Siruseri/team20/Srinidhi/AI Hack/Data/kb_docs.csv", source_column='text')
documents = loader.load()

# Embeddings
class EmbeddingManager:
    def __init__(self, model_path: str = 'Y:/Hackathon/Chennai/Siruseri/team20/Srinidhi/AI Hack/models/all-MiniLM-L6-v2.pt'):
        self.model_path = model_path
        self.model = None
        self._load_model() 

    def _load_model(self):
        try:
            print(f"Loading model: {self.model_path}")
            self.model = SentenceTransformer(self.model_path)
            print("Model loaded successfully.")
            print(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model is not loaded.")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings

    def get_embedding_dimension(self) -> int:
        if not self.model:
            raise ValueError("Model is not loaded.")
        return self.model.get_sentence_embedding_dimension()

embedding_manager = EmbeddingManager()

# VectorStore
class VectorStoreDB:
    def __init__(self, collection_name: str = "insurance_knowledge_base", persist_directory: str = "../Data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            print(f"ChromaDB collection '{self.collection_name}' initialized at '{self.persist_directory}'.")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise e

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match.")
        print(f"Adding {len(documents)} documents to the vector store...")

        ids=[]
        documents_texts=[]
        embeddings_list=[]
        metadatas=[]
        for i, doc in enumerate(zip(documents, embeddings)):
            doc_obj, embedding = doc
            unique_id = str(uuid.uuid4())
            ids.append(unique_id)
            documents_texts.append(doc_obj.page_content)
            embeddings_list.append(embedding.tolist())
            metadatas.append(doc_obj.metadata)
        try:
            self.collection.add(
                ids=ids,
                documents=documents_texts,
                embeddings=embeddings_list,
                metadatas=metadatas
            )
            print("Documents added successfully.")
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise e

vectorestore = VectorStoreDB()
vectorestore._initialize_store()

texts = [doc.page_content for doc in documents]
embeddings = embedding_manager.generate_embeddings(texts)
vectorestore.add_documents(documents, embeddings)

# RAG Retriever
class RAGRetriever:
    def __init__(self, vector_store: VectorStoreDB, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            "document": document,
                            "metadata": metadata if metadata else {},
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1
                        })
            return retrieved_docs
        except Exception as e:
            print(f"Error during retrieval: {e}")
            raise e

rag_retriever = RAGRetriever(vector_store=vectorestore, embedding_manager=embedding_manager)

# Load other data
customer_df = pd.read_csv("Y:/Hackathon/Chennai/Siruseri/team20/Srinidhi/AI Hack/Data/customer_crm.csv")
risk_df = pd.read_csv("Y:/Hackathon/Chennai/Siruseri/team20/Srinidhi/AI Hack/Data/risk_profile.csv")
product_df = pd.read_csv("Y:/Hackathon/Chennai/Siruseri/team20/Srinidhi/AI Hack/Data/product_catalog.csv")

# LLM setup
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in/",
    model="azure/genailab-maas-gpt-4o",
    api_key="sk-9Iuu7yap97VGk9mYCVa4pw",
    http_client=client
)

def rag_insurance(query, customer_id, retriever, llm, customer_df, risk_df, product_df, top_k=3):
    top_docs = retriever.retrieve(query, top_k=top_k)
    customer_profile = customer_df[customer_df['customer_id'] == customer_id].to_dict(orient='records')[0]
    risk_profile = risk_df[risk_df['customer_id'] == customer_id].to_dict(orient='records')[0]
    risk_segment = risk_profile['risk_segment']
    candidate_products = product_df[
        product_df['recommended_for_segments'].str.contains(risk_segment, case=False, na=False)
    ].to_dict(orient='records')

    prompt = f"""
You are an AI insurance assistant helping brokers recommend suitable products.

Customer Profile:
Name: {customer_profile['name']}
Age: {customer_profile['age']}
Gender: {customer_profile['gender']}
Occupation: {customer_profile['occupation']}
Income: {customer_profile['income_band']}
Existing Policies: {customer_profile['existing_policies']}
Claims History: {customer_profile['claims_history']}

Risk Profile:
Risk Segment: {risk_profile['risk_segment']}
Score: {risk_profile['score']}
Drivers: {risk_profile['drivers']}

Candidate Products:
"""
    for i, prod in enumerate(candidate_products):
        prompt += f"{i+1}. {prod['name']} ({prod['category']}) - {prod['key_features']}\n"

    prompt += "\nRelevant Knowledge Base Documents:\n"
    for i, doc in enumerate(top_docs):
        prompt += f"{i+1}. {doc['document']}\n"

    prompt += f"\nQuestion: {query}\nAnswer professionally, recommending suitable products based on the customer profile, risk profile, and knowledge base."

    response = llm([HumanMessage(content=prompt)])
    return response.content

# =========================
# Frontend UI (CoPilot Style)
# =========================

st.set_page_config(page_title="CoPilot for Insurance Brokerage Agencies", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
.main-title { font-size:2.8rem; font-weight:800; color:#0d47a1; margin-bottom:0.3rem; letter-spacing:2px; font-family:'Segoe UI',Arial,sans-serif; }
.subtitle { font-size:1.25rem; color:#1976d2; margin-bottom:2.2rem; font-family:'Segoe UI',Arial,sans-serif; }
.stButton > button { background: linear-gradient(90deg, #1976d2 0%, #64b5f6 100%); color:white; font-weight:700; border-radius:8px; padding:0.7rem 2rem; margin-top:1rem; font-size:1.1rem; box-shadow:0 2px 8px rgba(25, 118, 210, 0.08); border:none; transition: background 0.3s; }
.stButton > button:hover { background: linear-gradient(90deg, #1565c0 0%, #90caf9 100%); }
.stTextInput, .stTextArea, .stSelectbox { border-radius:8px; border:1.5px solid #1976d2; font-size:1.08rem; font-family:'Segoe UI',Arial,sans-serif; background-color:#f5faff; }
.stSidebar { background-color:#e3f2fd; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='width:100%;display:flex;align-items:center;justify-content:flex-start;background:linear-gradient(90deg,#1976d2 0%,#64b5f6 100%);padding:1.2em 2em 1.2em 1em;border-radius:12px;margin-bottom:1.5em;'>
<img src='https://cdn-icons-png.flaticon.com/512/3135/3135715.png' alt='Insurance CoPilot Logo' style='height:56px;width:56px;margin-right:1.5em;border-radius:8px;border:2px solid #fff;box-shadow:0 2px 8px rgba(25,118,210,0.12);'>
<span style='font-size:2.1rem;font-weight:800;color:#fff;letter-spacing:2px;font-family:Segoe UI,Arial,sans-serif;'>CoPilot for Insurance Brokerage Agencies</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Home", "Chatbot", "Customer Profile", "Sales Collateral", "About", "Contact"))
st.sidebar.subheader("Language Settings")
language = st.sidebar.selectbox("Choose your language", ["English", "Tamil", "Hindi", "French", "Spanish", "Chinese", "German"])
st.markdown("<div class='subtitle'>Empowering brokers with AI-driven customer engagement and sales</div>", unsafe_allow_html=True)

# ========================
# Chatbot Page
# ========================
if page == "Chatbot":
    st.header("Conversational Insurance Chatbot")
    if "history" not in st.session_state:
        st.session_state.history = []

    customer_ids = customer_df['customer_id'].tolist()
    selected_customer = st.selectbox("Select Customer", customer_ids)

    query = st.text_input("Type your question here:")
    if st.button("Send") and query:
        with st.spinner("Generating answer..."):
            answer = rag_insurance(
                query=query,
                customer_id=selected_customer,
                retriever=rag_retriever,
                llm=llm,
                customer_df=customer_df,
                risk_df=risk_df,
                product_df=product_df,
                top_k=3
            )
        st.session_state.history.append({"query": query, "answer": answer})

    for chat in st.session_state.history[::-1]:
        st.markdown(f"<div style='background:#e3f2fd;padding:0.7em 1em;border-radius:8px;margin-bottom:0.5em;color:#0d47a1;'><strong>Copilot:</strong> {chat['answer']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background:#fffde7;padding:0.7em 1em;border-radius:8px;margin-bottom:0.5em;color:#f57c00;'><strong>You:</strong> {chat['query']}</div>", unsafe_allow_html=True)

# ========================
# Customer Profile Page
# ========================
elif page == "Customer Profile":
    st.header("Customer Profile & Risk Analysis")
    customer_ids = customer_df['customer_id'].tolist()
    selected_customer = st.selectbox("Select Customer", customer_ids)
    customer_profile = customer_df[customer_df['customer_id'] == selected_customer].to_dict(orient='records')[0]
    risk_profile = risk_df[risk_df['customer_id'] == selected_customer].to_dict(orient='records')[0]

    st.subheader("Customer Details")
    st.json(customer_profile)
    st.subheader("Risk Profile")
    st.json(risk_profile)

# ========================
# Sales Collateral Page
# ========================
elif page == "Sales Collateral":
    st.header("Generate Personalized Insurance Sales Collateral")
    customer_ids = customer_df['customer_id'].tolist()
    selected_customer = st.selectbox("Select Customer", customer_ids)
    customer_profile = customer_df[customer_df['customer_id'] == selected_customer].to_dict(orient='records')[0]
    risk_profile = risk_df[risk_df['customer_id'] == selected_customer].to_dict(orient='records')[0]

    collateral_prompt = f"""
Generate a personalized sales collateral for this customer including:
Customer Profile: {json.dumps(customer_profile)}
Risk Profile: {json.dumps(risk_profile)}
Recommended Products: {json.dumps(product_df.to_dict(orient='records'))}
Respond in {language}.
"""
    if st.button("Generate Collateral"):
        with st.spinner("Generating collateral..."):
            collateral = rag_insurance(
                query=collateral_prompt,
                customer_id=selected_customer,
                retriever=rag_retriever,
                llm=llm,
                customer_df=customer_df,
                risk_df=risk_df,
                product_df=product_df,
                top_k=3
            )
        st.markdown(f"<div style='background:#e3f2fd;padding:1em;border-radius:8px;margin-bottom:0.5em;color:#0d47a1;'>{collateral}</div>", unsafe_allow_html=True)

# ========================
# About Page
# ========================
elif page == "About":
    st.header("About CoPilot for Insurance Brokerage Agencies")
    st.write("This AI agent helps brokers deliver personalized customer experiences, recommend products, and generate sales collateral using customer data, social feeds, and knowledge base documents.")

# ========================
# Contact Page
# ========================
elif page == "Contact":
    st.header("Contact")
    st.write("Contact us at: demo@example.com")
