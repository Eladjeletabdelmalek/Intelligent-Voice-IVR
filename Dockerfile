FROM python:3.12

WORKDIR /app
COPY . /app

# System dependencies for audio and building vLLM
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install qwen-asr with vLLM support
# Note: This might take a while as vLLM is a large package
RUN pip install --no-cache-dir fastapi uvicorn numpy "qwen-asr[vllm]" \
    langchain langchain-community langchain-google-genai \
    langchain-text-splitters chromadb pypdf

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]