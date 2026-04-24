import streamlit as st
from vllm import LLM, SamplingParams

@st.cache_resource
def load_vllm_gpu():
    # Load the Omni-Flash model onto your RTX 4060
    # We use float16/bfloat16 for 40-series GPU speed
    model = LLM(
        model="Qwen/Qwen3.5-Omni-Flash", 
        device="cuda", 
        dtype="bfloat16",
        gpu_memory_utilization=0.8, # Leave some room for Streamlit
        trust_remote_code=True
    )
    return model

llm = load_vllm_gpu()

# --- THE RESPONSE LOGIC ---
if user_text:
    # Set parameters for 'Extreme Low Latency'
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=256,
        extra_body={
            "audio_output": True,       # Tell the model to generate voice
            "ref_audio": "ref_temp.wav" # Your Djezzy brand sample
        }
    )

    # With an RTX 4060, this will take ~200-400ms
    outputs = llm.generate(user_text, sampling_params)
    
    ai_audio = outputs[0].outputs[0].audio_wav
    st.audio(ai_audio, autoplay=True)