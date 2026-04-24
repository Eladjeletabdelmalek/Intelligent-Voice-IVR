# 1. Use the official vLLM CUDA runtime
FROM vllm/vllm-openai:latest

WORKDIR /app

# 2. Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    ffmpeg \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Install requirements with the --ignore-installed flag
# This forces the install even if system-locked packages like 'blinker' exist
RUN pip install --no-cache-dir --ignore-installed \
    blinker \
    vllm-omni \
    streamlit \
    langchain \
    langchain-community \
    langchain-google-genai \
    chromadb \
    pypdf \
    soundfile

# 4. Copy files
COPY . /app

# 5. Expose ports
EXPOSE 8501

# 6. Run the app
CMD ["streamlit", "run", "vocal_test.py", "--server.address", "0.0.0.0"]










# FROM python:3.12-slim

# WORKDIR /app

# # Install system dependencies for audio and SQLite (required for ChromaDB)
# RUN apt-get update && apt-get install -y \
#     portaudio19-dev \
#     ffmpeg \
#     build-essential \
#     libsqlite3-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Install Python requirements
# # Note: Ensure you use the specific version of vLLM compatible with your CUDA 13.2/12.x drivers
# RUN pip install --no-cache-dir \
#     fastapi uvicorn numpy "qwen-asr[vllm]" \
#     langchain langchain-community langchain-google-genai \
#     langchain-text-splitters chromadb pypdf

# COPY . /app

# EXPOSE 8000

# CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]