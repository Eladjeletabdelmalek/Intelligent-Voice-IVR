import numpy as np
from fastapi import FastAPI, WebSocket
from qwen_asr import Qwen3ASRModel

app = FastAPI()

print("Loading model...")
asr = Qwen3ASRModel.LLM(
    model="Qwen/Qwen3-ASR-0.6B",
    gpu_memory_utilization=0.4,  # Lowered from 0.6
    max_new_tokens=32,
)
state = asr.init_streaming_state(
    unfixed_chunk_num=2,
    unfixed_token_num=5,
    chunk_size_sec=2.0,
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        data = await websocket.receive_bytes()
        wav_chunk = np.frombuffer(data, dtype=np.float32)

        asr.streaming_transcribe(wav_chunk, state)

        await websocket.send_text(state.text)