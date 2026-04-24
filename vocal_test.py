import streamlit as st
import torch
import os
from vllm import LLM, SamplingParams

st.set_page_config(page_title="Djezzy Omni-Turbo", page_icon="⚡")

# 1. Load the Unified Omni Model (GPU)
@st.cache_resource
def load_vllm_omni():
    # Qwen3.5-Omni-Flash fits perfectly in 8GB VRAM with bfloat16
    return LLM(
        model="Qwen/Qwen3.5-Omni-Flash",
        device="cuda",
        dtype="bfloat16",
        # Reserve 70% of 8GB for the model, leave rest for system/UI
        gpu_memory_utilization=0.7, 
        trust_remote_code=True
    )

llm = load_vllm_omni()

st.title("🎙️ Djezzy Omni-Bot (RTX 4060)")

with st.sidebar:
    st.header("⚙️ Brand Settings")
    ref_file = st.file_uploader("Upload Brand Voice (3-5s):", type=['wav', 'mp3'])
    st.info("Using GPU-accelerated Zero-Shot Cloning.")

# --- INTERACTION ---
user_audio = st.audio_input("Tap to talk to Djezzy")

if user_audio and ref_file:
    # Save files for vLLM to pick up
    with open("user_voice.wav", "wb") as f:
        f.write(user_audio.read())
    with open("brand_voice.wav", "wb") as f:
        f.write(ref_file.getvalue())

    with st.spinner("🤖 Processing..."):
        # The 'Omni' magic: we pass the audio file directly. 
        # No separate Whisper needed!
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=256,
            extra_body={
                "audio_output": True,        # Ask for vocal response
                "ref_audio": "brand_voice.wav", 
                "modality": "audio"          # Process input as audio
            }
        )

        # Generate response using GPU
        # In Omni models, prompt can be the file path to audio
        outputs = llm.generate({"prompt": "user_voice.wav", "multi_modal_data": {"audio": "user_voice.wav"}}, sampling_params)
        
        # Extract Text and Audio
        ai_text = outputs[0].outputs[0].text
        ai_audio = outputs[0].outputs[0].extra_data.get("audio_wav")

    # --- DISPLAY ---
    st.chat_message("user").write("*(Vocal Input)*")
    
    with st.chat_message("assistant"):
        st.write(ai_text)
        if ai_audio:
            st.audio(ai_audio, autoplay=True)

elif user_audio and not ref_file:
    st.warning("Please upload the brand voice first!")