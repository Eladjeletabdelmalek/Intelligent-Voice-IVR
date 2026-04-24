import streamlit as st
import torch
import os
import soundfile as sf
from vllm import LLM, SamplingParams

# LangChain / RAG Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. INITIAL CONFIG ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyC24SxD24zLNDVB7CyP28DJKQPLPX-7AgA" # Replace with your key
st.set_page_config(page_title="Djezzy Omni-RAG", page_icon="🎙️", layout="wide")

# --- 2. THE BRAIN (RAG Logic) ---
@st.cache_resource
def load_rag_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
    llm_brain = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.3)
    
    # Ensure this directory exists with your Djezzy PDFs indexed
    vectorstore = Chroma(persist_directory='./chroma_db', embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Contextualize Question
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", """Rephrase the user question into a standalone question based on history."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm_brain, retriever, context_prompt)

    # QA Chain
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Djezzy assistant. Use the context to answer in a helpful, concise way.
        If the answer is not in the context, Try to respond in general like a normal chat person . 
        If he changes his language or speak an other language, answer in that language .
         \n\nContext: {context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm_brain, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, document_chain)

# --- 3. THE VOICE (vLLM Omni) ---
@st.cache_resource
def load_vllm_omni():
    # Loading on RTX 4060 (8GB) - Using 0.7 to leave room for RAG/UI
    return LLM(
        model="Qwen/Qwen3.5-Omni-Flash",
        device="cuda",
        dtype="bfloat16",
        gpu_memory_utilization=0.7, 
        trust_remote_code=True
    )

# Initialize Session State for Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load everything
rag_chain = load_rag_chain()
omni_model = load_vllm_omni()

# --- 4. UI LAYOUT ---
st.title("🎙️ Djezzy Smart IVR Assistant")
st.caption("Powered by Qwen3.5-Omni & Gemini RAG")

with st.sidebar:
    st.header("⚙️ Configuration")
    ref_file = st.file_uploader("Upload Brand Voice (3-5s):", type=['wav', 'mp3'])
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()

# --- 5. THE INTERACTION LOOP ---
user_audio = st.audio_input("Ask Djezzy about plans, SIMs, or support")

if user_audio and ref_file:
    # A. Save audio locally for vLLM to read
    with open("user_input.wav", "wb") as f:
        f.write(user_audio.read())
    with open("brand_ref.wav", "wb") as f:
        f.write(ref_file.getvalue())

    with st.spinner("Djezzy is thinking..."):
        # STEP 1: UNDERSTAND (Audio-to-Text via Omni)
        # We use the Omni model to transcribe the audio first
        transcription_params = SamplingParams(temperature=0.0, max_tokens=128)
        # In 2026 Omni models, providing audio modality usually defaults to transcription if not asked otherwise
        ts_output = omni_model.generate(
            {"prompt": "<|audio|>Describe this audio.", "multi_modal_data": {"audio": "user_input.wav"}}, 
            transcription_params
        )
        user_text = ts_output[0].outputs[0].text
        
        # STEP 2: REASON (RAG via Gemini)
        # Use the transcription to query your Djezzy PDF database
        response = rag_chain.invoke({
            "input": user_text, 
            "chat_history": st.session_state.chat_history
        })
        answer_text = response["answer"]

        # STEP 3: SPEAK (Text-to-Audio via Omni)
        # Convert the RAG answer back into the brand's voice
        vocal_params = SamplingParams(
            temperature=0.7,
            extra_body={
                "audio_output": True,
                "ref_audio": "brand_ref.wav"
            }
        )
        vocal_output = omni_model.generate(answer_text, vocal_params)
        ai_audio = vocal_output[0].outputs[0].extra_data.get("audio_wav")

    # --- 6. DISPLAY RESULTS ---
    st.chat_message("user").write(user_text)
    
    with st.chat_message("assistant"):
        st.write(answer_text)
        if ai_audio:
            # Note: vLLM returns raw wav data, we might need to save/play
            st.audio(ai_audio, autoplay=True)
            
    # Update Memory
    st.session_state.chat_history.extend([
        HumanMessage(content=user_text),
        AIMessage(content=answer_text)
    ])

elif user_audio and not ref_file:
    st.warning("Please upload a brand voice sample in the sidebar.")

















# import os
# import time
# import numpy as np
# from fastapi import FastAPI, WebSocket, HTTPException
# from pydantic import BaseModel
# from qwen_asr import Qwen3ASRModel

# # LangChain / RAG Imports
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.vectorstores import Chroma
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# # --- 1. CONFIGURATION & MODELS ---
# os.environ["GOOGLE_API_KEY"] = "AIzaSyBnexXZFlr2tuvQCCIfcG-1oLC85HHHdIU" # Note: Keep this key safe!

# app = FastAPI(title="Djezzy-IVR-Smart-ASR")

# # Load Qwen (The "Ear")
# print("Loading Qwen3-ASR-0.6B...")
# asr = Qwen3ASRModel.LLM(
#     model="Qwen/Qwen3-ASR-0.6B",
#     gpu_memory_utilization=0.7, # Lowered slightly to leave room for other processes
#     enforce_eager=True,
#     max_model_len=2048,
#     trust_remote_code=True
# )

# # Load RAG Chain (The "Brain")
# print("Connecting to Knowledge Base...")
# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
# llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.3)
# vectorstore = Chroma(persist_directory='./chroma_db', embedding_function=embeddings)
# retriever = vectorstore.as_retriever()

# # Build the RAG Logic
# contextualize_q_prompt = ChatPromptTemplate.from_messages([
#     ("system", "Rephrase the user question into a standalone question based on history."),
#     MessagesPlaceholder("chat_history"),
#     ("human", "{input}"),
# ])
# history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# qa_prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a Djezzy assistant. Use the context to answer.\n\n{context}"),
#     MessagesPlaceholder("chat_history"),
#     ("human", "{input}"),
# ])
# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
# rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# # Memory for the API (In a real app, this should be per-user)
# global_chat_history = []

# # --- 2. ENDPOINTS ---

# class TextRequest(BaseModel):
#     prompt: str

# @app.post("/ask")
# async def ask_knowledge_base(request: TextRequest):
#     """Answers text questions using the PDF knowledge base."""
#     global global_chat_history
#     try:
#         response = rag_chain.invoke({
#             "input": request.prompt, 
#             "chat_history": global_chat_history
#         })
        
#         # Update History
#         global_chat_history.append(HumanMessage(content=request.prompt))
#         global_chat_history.append(AIMessage(content=response["answer"]))
        
#         return {"answer": response["answer"]}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     """Real-time ASR that eventually triggers RAG."""
#     await websocket.accept()
#     state = asr.init_streaming_state(unfixed_chunk_num=2, unfixed_token_num=5, chunk_size_sec=2.0)
    
#     try:
#         while True:
#             data = await websocket.receive_bytes()
#             wav_chunk = np.frombuffer(data, dtype=np.float32)
#             asr.streaming_transcribe(wav_chunk, state)
            
#             # Send live transcription back
#             await websocket.send_text(f"Processing: {state.text}")
            
#             # Logic: If user stops talking, you'd trigger rag_chain.invoke(state.text)
#     except Exception as e:
#         print(f"WS Error: {e}")
#     finally:
#         await websocket.close()
        