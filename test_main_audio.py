"""语音对话版本：集成 ASR、Agent、TTS 的完整流程

核心流程：
1) Omni 进行麦克风拾音 + VAD + ASR（语音转文本）
2) ASR 完成一句话后，将文本发送给后台处理线程
3) 后台线程调用 agent.handle_message(text, db) 进行文本2文本处理
4) 生成的回复通过 TTS 合成为语音并播放

用法示例：
    python test_main_audio.py
启动后说话即可。输入 exit 或按 Ctrl+C 退出。

所有关键步骤都会打印日志便于调试。
🎤 麦克风 → ASR (Omni)
    ↓
📝 文本识别完成
    ↓
📤 入队处理
    ↓
🤖 Agent.handle_message() 
    ↓
💬 生成文本回复
    ↓
🔊 TTS 合成并播放
"""

import base64
import os
import queue
import signal
import sys
import threading
import time
from typing import Any, Optional

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

from B64PCMPlayer import B64PCMPlayer
from src.db import Base, engine, SessionLocal
from src.llm_client import LLMClient
from src.rag_service import RAGService
from src.plan_service import PlanService
from src.quiz_service import QuizService
from src.review_service import ReviewService
from src.vector_store import MilvusVectorStore
from src.chat_agent import PlanChatAgent
from src.quiz_chat_agent import QuizChatAgent
from src.main_chat_agent import MainChatAgent


# === Configuration ===
ASR_MODEL = "qwen-omni-turbo-realtime-latest"
ASR_TRANSCRIPTION_MODEL = "gummy-realtime-v1"
TTS_MODEL = "qwen3-tts-flash-realtime"
TTS_VOICE = "Cherry"


def init_dashscope_api_key():
    """初始化 DashScope API 密钥"""
    if "DASHSCOPE_API_KEY" in os.environ:
        dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
    else:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY ")


def init_agent() -> MainChatAgent:
    """初始化 MainChatAgent，与 test_main.py 一致"""
    print("[AGENT] 初始化 Agent 依赖...")
    Base.metadata.create_all(bind=engine)

    vec = MilvusVectorStore()
    llm = LLMClient()
    rag = RAGService(vec, llm)
    plans = PlanService(vec, llm)
    reviews = ReviewService()
    quizzes = QuizService(vec, llm, reviews)
    plan_chat = PlanChatAgent(plans, llm)
    quiz_chat = QuizChatAgent(quizzes, llm)
    agent = MainChatAgent(llm, rag, plan_chat, quiz_chat, reviews)
    print("[AGENT] Agent 初始化完成")
    return agent


# === Global runtime objects ===
task_queue: "queue.Queue[str]" = queue.Queue()
stop_event = threading.Event()
mic_stream = None
player: Optional[B64PCMPlayer] = None
conversation: Optional[OmniRealtimeConversation] = None
agent: Optional[MainChatAgent] = None


class AsrCallback(OmniRealtimeCallback):
    """ASR 回调处理：仅负责语音识别"""

    def on_open(self) -> None:
        global mic_stream
        print("[ASR] 连接已打开，初始化麦克风...")
        pya = pyaudio.PyAudio()
        mic_stream = pya.open(
            format=pyaudio.paInt16, channels=1, rate=16000, input=True
        )

    def on_close(self, close_status_code, close_msg) -> None:
        print(f"[ASR] 连接已关闭 code={close_status_code}, msg={close_msg}")

    def on_event(self, response: Any) -> None:
        try:
            evt_type = response.get("type") if isinstance(response, dict) else None

            if evt_type == "session.created":
                session_id = response.get("session", {}).get("id")
                print(f"[ASR] 会话已创建: {session_id}")

            if evt_type == "input_audio_buffer.speech_started":
                print("[ASR] VAD: 检测到语音开始，如有播放中则停止")
                if player:
                    player.cancel_playing()

            if evt_type == "conversation.item.input_audio_transcription.completed":
                transcript = response.get("transcript", "")
                print(f"[ASR] 语音识别完成: '{transcript}'")
                if transcript.strip():
                    print("[QUEUE] 将文本入队处理")
                    task_queue.put(transcript)

        except Exception as exc:
            print(f"[ASR] 错误: {exc}")


class TtsCallback(QwenTtsRealtimeCallback):
    """TTS 回调处理：合成并播放语音"""

    def __init__(self, player: B64PCMPlayer):
        self.player = player
        self.done_event = threading.Event()

    def on_open(self) -> None:
        print("[TTS] 连接已打开")

    def on_close(self, close_status_code, close_msg) -> None:
        print(f"[TTS] 连接已关闭 code={close_status_code}, msg={close_msg}")
        self.done_event.set()

    def on_event(self, response: Any) -> None:
        try:
            evt_type = response.get("type") if isinstance(response, dict) else None

            if evt_type == "session.created":
                session_id = response.get("session", {}).get("id")
                print(f"[TTS] 会话已创建: {session_id}")

            if evt_type == "response.audio.delta":
                audio_b64 = response.get("delta")
                if audio_b64:
                    self.player.add_data(audio_b64)

            if evt_type == "response.done":
                print("[TTS] 合成完成，播放结束")
                self.done_event.set()

        except Exception as exc:
            print(f"[TTS] 错误: {exc}")

    def wait_done(self):
        self.done_event.wait()


def synthesize_and_play(text: str, pya: pyaudio.PyAudio, player_ref: B64PCMPlayer):
    """合成文本为语音并播放"""
    print(f"[TTS] 开始合成语音: {text}")
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

    # 推送文本并完成
    tts_client.append_text(text)
    tts_client.finish()
    tts_callback.wait_done()
    print("[TTS] 播放完成")


def worker_process_message(pya: pyaudio.PyAudio):
    """后台处理线程：调用 Agent 处理文本，然后合成并播放回复

    流程：
    1) 从队列取出 ASR 识别的文本
    2) 调用 agent.handle_message(text, db) 进行处理
    3) 获取回复并通过 TTS 合成播放
    """
    while not stop_event.is_set():
        try:
            text = task_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if agent is None:
            print("[WORKER] 错误：Agent 未初始化")
            task_queue.task_done()
            continue

        print(f"[WORKER] 处理文本: '{text}'")

        try:
            with SessionLocal() as db:
                print("[AGENT] 调用 agent.handle_message()...")
                result = agent.handle_message(text, db)

            reply = result.get("reply") if isinstance(result, dict) else result
            intent = result.get("intent") if isinstance(result, dict) else None

            if intent:
                print(f"[AGENT] 回复 [intent={intent}]: {reply}")
            else:
                print(f"[AGENT] 回复: {reply}")

            # 合成并播放回复
            if reply and player:
                synthesize_and_play(reply, pya, player)

        except Exception as exc:
            print(f"[WORKER] 处理错误: {exc}")

        task_queue.task_done()


def signal_handler(sig, frame):
    """处理 Ctrl+C 信号"""
    print("\n[SYS] 收到退出信号，正在关闭...")
    stop_event.set()
    if conversation:
        conversation.close()
    if player:
        player.shutdown()
    print("[SYS] 再见！")
    sys.exit(0)


def main():
    """主程序入口"""
    global agent, player, conversation

    try:
        init_dashscope_api_key()
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # 初始化 Agent
    agent = init_agent()

    signal.signal(signal.SIGINT, signal_handler)

    print("[SYS] 初始化语音对话流程...")
    pya = pyaudio.PyAudio()
    player = B64PCMPlayer(pya, sample_rate=24000, chunk_size_ms=80)

    # 初始化 ASR
    print("[SYS] 初始化 ASR (Omni)...")
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

    # 启动后台处理线程
    print("[SYS] 启动后台处理线程...")
    worker = threading.Thread(target=worker_process_message, args=(pya,), daemon=True)
    worker.start()

    print("[SYS] 语音对话系统已就绪。请对着麦克风说话。按 Ctrl+C 退出。\n")

    # 主循环：持续读取麦克风音频
    last_log = time.time()
    try:
        while True:
            if mic_stream is None:
                time.sleep(0.05)
                continue

            audio_data = mic_stream.read(3200, exception_on_overflow=False)
            audio_b64 = base64.b64encode(audio_data).decode("ascii")
            conversation.append_audio(audio_b64)

            # 定期打印队列状态
            if time.time() - last_log > 5:
                queue_size = task_queue.qsize()
                print(f"[SYS] ASR 流式处理中... 队列大小={queue_size}")
                last_log = time.time()

    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()
