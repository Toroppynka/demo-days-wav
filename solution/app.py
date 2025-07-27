"""FastAPI сервис, кодирующий текст в WAV и декодирующий его обратно.

✦ Реализуйте функции `text_to_audio` и `audio_to_text`.
✦ Формат аудио: 44100Hz, 16‑bit PCM, mono.
"""

import base64
import io
import wave

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

SAMPLE_RATE = 44_100   # Hz
BIT_DEPTH = 16         # bits per sample
CHANNELS = 1
SYMBOL_DURATION = 0.1  # seconds

# Частоты, соответствующие цифрам 0-9
FREQ_MAP = {
    '0': 500,
    '1': 600,
    '2': 700,
    '3': 800,
    '4': 900,
    '5': 1000,
    '6': 1100,
    '7': 1200,
    '8': 1300,
    '9': 1400,
}
INV_FREQ_MAP = {v: k for k, v in FREQ_MAP.items()}
FREQ_LIST = list(FREQ_MAP.values())

app = FastAPI(swagger_ui_parameters={"syntaxHighlight": False})

# ---------------------------- pydantic models ---------------------------- #

class EncodeRequest(BaseModel):
    text: str = Field(..., description="Строка для кодирования в звук")


class EncodeResponse(BaseModel):
    data: str  # base64 wav


class DecodeRequest(BaseModel):
    data: str  # base64 wav


class DecodeResponse(BaseModel):
    text: str


# ---------------------------- helpers ---------------------------- #

def _empty_wav(duration_sec: float = 1.0) -> bytes:
    """Возвращает WAV‑байты тишины длиной *duration_sec*."""
    n_samples = int(SAMPLE_RATE * duration_sec)
    silence = np.zeros(n_samples, dtype=np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(BIT_DEPTH // 8)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(silence.tobytes())
    return buf.getvalue()


def _generate_tone(freq: float, duration: float) -> np.ndarray:
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave_data = np.sin(2 * np.pi * freq * t)
    return wave_data


def _detect_freq(chunk: np.ndarray) -> str:
    spectrum = np.fft.fft(chunk)
    freqs = np.fft.fftfreq(len(chunk), 1 / SAMPLE_RATE)
    magnitude = np.abs(spectrum[:len(spectrum)//2])
    freqs = freqs[:len(freqs)//2]
    peak_freq = freqs[np.argmax(magnitude)]
    closest = min(FREQ_LIST, key=lambda x: abs(x - peak_freq))
    return INV_FREQ_MAP[closest]

# ---------------------------- main logic ---------------------------- #

def text_to_audio(text: str) -> bytes:
    tones = []
    for ch in text:
        freq = FREQ_MAP.get(ch, 500)  # по умолчанию 500 Гц, если символ неизвестен
        tone = _generate_tone(freq, SYMBOL_DURATION)
        tones.append(tone)

    audio = np.concatenate(tones)
    audio *= 32767 / np.max(np.abs(audio))
    audio = audio.astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(BIT_DEPTH // 8)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def audio_to_text(wav_bytes: bytes) -> str:
    with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

    chunk_size = int(SAMPLE_RATE * SYMBOL_DURATION)
    text = ""
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < chunk_size:
            break
        text += _detect_freq(chunk)

    return text


# ---------------------------- endpoints ---------------------------- #

@app.post("/encode", response_model=EncodeResponse)
async def encode_text(request: EncodeRequest):
    wav_bytes = text_to_audio(request.text)
    wav_base64 = base64.b64encode(wav_bytes).decode("utf-8")
    return EncodeResponse(data=wav_base64)


@app.post("/decode", response_model=DecodeResponse)
async def decode_audio(request: DecodeRequest):
    wav_bytes = base64.b64decode(request.data)
    text = audio_to_text(wav_bytes)
    return DecodeResponse(text=text)


@app.get("/ping")
async def ping():
    return "ok"
