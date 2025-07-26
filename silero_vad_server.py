
import asyncio
import websockets
import numpy as np
import onnxruntime as ort
import struct
import pdb
class SileroVAD:
    def __init__(self, model_path="./silero_vad.onnx", sample_rate=16000, window_ms=30):
        self.sr = sample_rate
        self.win = int(sample_rate * window_ms / 1000)
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.inp_names = [inp.name for inp in self.sess.get_inputs()]

    def is_speech(self, audio: np.ndarray, thresh: float = 0.5) -> bool:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if len(audio) < self.win:
            buf = np.zeros(self.win, dtype=np.float32)
            buf[-len(audio):] = audio
            audio = buf
        chunk = audio[-self.win:]
        inp = chunk.reshape(1, -1)
        feed = {"input": inp}
        if "sr" in self.inp_names:
            feed["sr"] = np.array([self.sr], dtype=np.int64)
        if "state" in self.inp_names:
            meta = next(i for i in self.sess.get_inputs() if i.name == "state")
            shape = [d or 1 for d in meta.shape]
            feed["state"] = np.zeros(shape, dtype=np.float32)
            # pdb.set_trace()
        prob = float(self.sess.run(None, feed)[0].item())
        return prob >= thresh


vad = SileroVAD()

async def vad_server(websocket, path=None):
    print("Client connected")
    buffer = np.array([], dtype=np.float32)

    try:
        async for message in websocket:
            int16_data = np.frombuffer(message, dtype=np.int16)
            float_data = int16_data.astype(np.float32) / 32768.0  # normalize
            buffer = np.concatenate((buffer, float_data))

            # only keep the last second of audio
            if len(buffer) > vad.sr:
                buffer = buffer[-vad.sr:]

            is_speaking = vad.is_speech(buffer)
            await websocket.send(f'{{"vad": {str(is_speaking).lower()}}}')
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    print("Starting WebSocket VAD server at ws://localhost:8765")
    async with websockets.serve(vad_server, "localhost", 8765):
        await asyncio.Future()  # run forever

asyncio.run(main())
