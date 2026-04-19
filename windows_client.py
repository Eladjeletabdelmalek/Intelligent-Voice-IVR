import pyaudio
import asyncio
import websockets
import numpy as np

RATE = 16000
CHUNK = int(RATE * 0.5)

async def stream_audio():
    uri = "ws://localhost:8000/ws"

    async with websockets.connect(uri) as ws:
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        print("Streaming...")

        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            await ws.send(data)

            text = await ws.recv()
            print(f"\r{text}", end="")

asyncio.run(stream_audio())