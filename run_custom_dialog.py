"""
Custom pipeline demo that decouples ASR (Omni), LLM (Qwen3.5 non-stream),
and TTS (Qwen TTS Realtime).

Pipeline:
1) Omni does only microphone capture + VAD + transcription.
2) After VAD completes a sentence, transcript is queued to a worker thread.
3) Worker calls non-stream Qwen3.5 to generate a reply.
4) Reply is synthesized to speech via Qwen TTS Realtime and played locally.

All important stages log to stdout for monitoring.
"""

import base64
import os
import queue
import signal
import sys
import threading
import time
from typing import Optional, Any

import dashscope
import pyaudio
from dashscope.audio.qwen_omni import (
    AudioFormat,
    MultiModality,
    OmniRealtimeCallback,
    OmniRealtimeConversation,
)
from dashscope.audio.qwen_tts_realtime import (
    AudioFormat as TtsAudioFormat,
    QwenTtsRealtime,
    QwenTtsRealtimeCallback,
)
from openai import OpenAI

from B64PCMPlayer import B64PCMPlayer


# === Configuration ===
ASR_MODEL = "qwen-omni-turbo-realtime-latest"
ASR_TRANSCRIPTION_MODEL = "gummy-realtime-v1"
LLM_MODEL = "qwen3.5-flash"
TTS_MODEL = "qwen3-tts-flash-realtime"
TTS_VOICE = "Cherry"


def init_dashscope_api_key():
    if "DASHSCOPE_API_KEY" in os.environ:
        dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
    else:
        raise ValueError(
            "Please set environment variable DASHSCOPE_API_KEY before running "
            "this demo."
        )


def build_llm_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "Please set environment variable DASHSCOPE_API_KEY for LLM client"
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


# === Global runtime objects ===
task_queue: "queue.Queue[str]" = queue.Queue()
stop_event = threading.Event()
mic_stream = None
player: Optional[B64PCMPlayer] = None
conversation: Optional[OmniRealtimeConversation] = None
llm_client: Optional[OpenAI] = None


class AsrCallback(OmniRealtimeCallback):
    def on_open(self) -> None:
        global mic_stream
        print("[ASR] connection opened, init microphone ...")
        pya = pyaudio.PyAudio()
        mic_stream = pya.open(
            format=pyaudio.paInt16, channels=1, rate=16000, input=True
        )

    def on_close(self, close_status_code, close_msg) -> None:
        print(f"[ASR] connection closed code={close_status_code}, " f"msg={close_msg}")

    def on_event(self, response: Any) -> None:
        try:
            evt_type = response.get("type") if isinstance(response, dict) else None
            if evt_type == "session.created":
                session_id = response.get("session", {}).get("id")
                print(f"[ASR] session created: {session_id}")
            if evt_type == "input_audio_buffer.speech_started":
                print("[ASR] VAD: speech started, stop playback if any")
                if player:
                    player.cancel_playing()
            if evt_type == "conversation.item.input_audio_transcription.completed":
                transcript = response.get("transcript", "")
                print(f"[ASR] transcript completed: {transcript}")
                task_queue.put(transcript)
        except Exception as exc:
            print(f"[ASR][Error] {exc}")


class TtsCallback(QwenTtsRealtimeCallback):
    def __init__(self, player: B64PCMPlayer):
        self.player = player
        self.done_event = threading.Event()

    def on_open(self) -> None:
        print("[TTS] connection opened")

    def on_close(self, close_status_code, close_msg) -> None:
        print(f"[TTS] connection closed code={close_status_code}, " f"msg={close_msg}")
        self.done_event.set()

    def on_event(self, response: Any) -> None:
        try:
            evt_type = response.get("type") if isinstance(response, dict) else None
            if evt_type == "session.created":
                session_id = response.get("session", {}).get("id")
                print(f"[TTS] session created: {session_id}")
            if evt_type == "response.audio.delta":
                audio_b64 = response.get("delta")
                if audio_b64:
                    self.player.add_data(audio_b64)
            if evt_type == "response.done":
                print("[TTS] response done")
                self.done_event.set()
        except Exception as exc:
            print(f"[TTS][Error] {exc}")

    def wait_done(self):
        self.done_event.wait()


def call_llm(client: OpenAI, query: str) -> Optional[str]:
    print(f"[LLM] start call, input='{query}'")
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个简洁的语音助手，请用口语化中文回答。",
                },
                {"role": "user", "content": query},
            ],
        )
        reply = completion.choices[0].message.content
        print(f"[LLM] reply: {reply}")
        return reply
    except Exception as exc:
        print(f"[LLM][Error] {exc}")
        return None


def synthesize_and_play(text: str, pya: pyaudio.PyAudio, player_ref: B64PCMPlayer):
    print(f"[TTS] start synthesis: {text}")
    tts_callback = TtsCallback(player_ref)
    tts_client = QwenTtsRealtime(
        model=TTS_MODEL,
        callback=tts_callback,
        url="wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
    )
    tts_client.connect()
    tts_client.update_session(
        voice=TTS_VOICE,
        response_format=TtsAudioFormat.PCM_24000HZ_MONO_16BIT,
        mode="server_commit",
    )

    # push text and finish
    tts_client.append_text(text)
    tts_client.finish()
    tts_callback.wait_done()
    print("[TTS] playback finished")


def llm_tts_worker(pya: pyaudio.PyAudio):
    while not stop_event.is_set():
        try:
            text = task_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if llm_client is None:
            print("[LLM][Error] client not initialized")
            continue
        reply = call_llm(llm_client, text)
        if reply:
            if player:
                synthesize_and_play(reply, pya, player)
        task_queue.task_done()


def signal_handler(sig, frame):
    print("[SYS] Ctrl+C pressed, stopping ...")
    stop_event.set()
    if conversation:
        conversation.close()
    if player:
        player.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    init_dashscope_api_key()
    llm_client = build_llm_client()

    signal.signal(signal.SIGINT, signal_handler)

    print("[SYS] Initializing custom dialog pipeline ...")
    pya = pyaudio.PyAudio()
    player = B64PCMPlayer(pya, sample_rate=24000, chunk_size_ms=80)

    callback = AsrCallback()
    conversation = OmniRealtimeConversation(model=ASR_MODEL, callback=callback)
    conversation.connect()
    conversation.update_session(
        output_modalities=[MultiModality.TEXT],
        voice="Chelsie",
        input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
        output_audio_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
        enable_input_audio_transcription=True,
        input_audio_transcription_model=ASR_TRANSCRIPTION_MODEL,
        enable_turn_detection=True,
        turn_detection_type="server_vad",
    )

    worker = threading.Thread(target=llm_tts_worker, args=(pya,), daemon=True)
    worker.start()

    print("[SYS] Ready. Speak into the microphone. Press Ctrl+C to exit.")
    last_log = time.time()
    while True:
        if mic_stream is None:
            time.sleep(0.05)
            continue
        audio_data = mic_stream.read(3200, exception_on_overflow=False)
        audio_b64 = base64.b64encode(audio_data).decode("ascii")
        conversation.append_audio(audio_b64)
        if time.time() - last_log > 5:
            print(f"[ASR] streaming audio ... queue size={task_queue.qsize()}")
            last_log = time.time()
