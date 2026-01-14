# -*- coding: utf-8 -*-
'''
@File    :   difyASR.py
@Author  :   一力辉
'''

from ..builder import ASREngines
from ..engineBase import BaseASREngine
import io, base64, json, re, time
from digitalHuman.protocol import AudioMessage, TextMessage, AUDIO_TYPE
from digitalHuman.utils import logger, httpxAsyncClient, wavToMp3

__all__ = ["DifyApiAsr"]


@ASREngines.register("Dify")
class DifyApiAsr(BaseASREngine):
    def setup(self):
        """从配置文件读取参数默认值并设置到实例属性"""
        for param in self.parameters():
            setattr(self, param.name, param.default)
        logger.info(f"[ASR] DifyApiAsr setup with wake_words={getattr(self, 'wake_words', 'N/A')}")

    def _parse_wake_words(self, wake_words_param):
        if isinstance(wake_words_param, str):
            return [w.strip() for w in re.split(r"[，,]", wake_words_param) if w.strip()]
        if isinstance(wake_words_param, list):
            return [str(w).strip() for w in wake_words_param if str(w).strip()]
        return []

    def _init_wake_config(self, **kwargs):
        wake_words = kwargs.get("wake_words") or getattr(self, "wake_words", "小木小木")
        self._wake_words = self._parse_wake_words(wake_words)
        self._auto_sleep = bool(kwargs.get("auto_sleep", getattr(self, "auto_sleep", False)))
        self._auto_sleep_seconds = int(kwargs.get("auto_sleep_seconds", getattr(self, "auto_sleep_seconds", 60)))
        if not hasattr(self, "_session_state"):
            self._session_state = {}  # {session_id: {"awake": bool, "last_ts": float}}
        logger.debug(f"[ASR Wake] Initialized with wake_words={wake_words}, parsed={self._wake_words}, auto_sleep={self._auto_sleep}, timeout={self._auto_sleep_seconds}")

    def _apply_wake_gate(self, session_id: str, text: str) -> str:
        now = time.time()
        state = self._session_state.get(session_id, {"awake": False, "last_ts": 0.0})

        # 超时自动休眠
        if state["awake"] and self._auto_sleep_seconds > 0 and now - state["last_ts"] > self._auto_sleep_seconds:
            state["awake"] = False

        # 检测唤醒词（不从文本中删除）
        matched = False
        for w in self._wake_words:
            if w and w in text:
                matched = True
                state["awake"] = True
                break

        logger.debug(f"[ASR Wake] Checking text='{text}', wake_words={self._wake_words}, matched={matched}, state={state}")

        # 未唤醒则返回空字符串（前端用于判断是否调用LLM）
        if not state["awake"] and not matched:
            self._session_state[session_id] = state
            return ""  # 返回空表示不调用LLM，但完整文本仍会显示在前端

        # 已唤醒，更新时间，返回完整文本
        state["last_ts"] = now

        # 每句后立即休眠（可选）
        if self._auto_sleep:
            state["awake"] = False

        self._session_state[session_id] = state
        return text  # 返回完整文本，不截断唤醒词

    async def run(self, input: AudioMessage, **kwargs) -> TextMessage:
        self._init_wake_config(**kwargs)

        # 参数校验
        paramters = self.checkParameter(**kwargs)
        API_SERVER = paramters["api_server"]
        API_KEY = paramters["api_key"]
        API_USERNAME = paramters["username"]

        headers = {
            'Authorization': f'Bearer {API_KEY}'
        }

        # 处理音频数据
        if isinstance(input.data, str):
            audio_data = base64.b64decode(input.data)
        else:
            audio_data = input.data

        if input.type == AUDIO_TYPE.WAV:
            audio_data = wavToMp3(audio_data)
            file_ext = "mp3"
        else:
            file_ext = "mp3"

        # 第一步：上传文件到 Dify
        upload_files = {
            "file": (f"audio.{file_ext}", io.BytesIO(audio_data), f"audio/{file_ext}")
        }
        upload_data = {
            "user": API_USERNAME
        }

        upload_response = await httpxAsyncClient.post(
            API_SERVER + "/files/upload",
            headers=headers,
            files=upload_files,
            data=upload_data
        )

        if upload_response.status_code != 200 and upload_response.status_code != 201:
            logger.error(f"[ASR] Dify file upload error: {upload_response.text}")
            raise RuntimeError(f"Dify file upload error: {upload_response.status_code}, detail: {upload_response.text}")

        upload_result = upload_response.json()
        file_id = upload_result.get("id")
        logger.debug(f"[ASR] File uploaded, id: {file_id}")

        # 第二步：调用工作流
        workflow_headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }

        workflow_payload = {
            "inputs": {
                "x": {
                    "transfer_method": "local_file",
                    "upload_file_id": file_id,
                    "type": "audio"
                }
            },
            "response_mode": "blocking",
            "user": API_USERNAME
        }

        response = await httpxAsyncClient.post(
            API_SERVER + "/workflows/run",
            headers=workflow_headers,
            json=workflow_payload
        )

        if response.status_code != 200:
            logger.error(f"[ASR] Dify API error response: {response.text}")
            raise RuntimeError(f"Dify asr api error: {response.status_code}, detail: {response.text}")

        result_data = response.json()
        # 从工作流输出中获取 text 字段
        result = result_data.get("data", {}).get("outputs", {}).get("text", "")

        logger.debug(f"[ASR] Engine response: {result}")

        session_id = kwargs.get("session_id") or kwargs.get("stream_id") or kwargs.get("uuid") or "default"
        full_text = result  # 保存完整文本用于前端显示
        gated = self._apply_wake_gate(session_id, result)  # 过滤后的文本供 LLM 使用

        logger.info(f"[ASR Debug] session_id={session_id}, full_text='{full_text}', gated='{gated}'")
        
        return TextMessage(
            data=gated,
            metadata={"full_text": full_text} if full_text != gated else None
        )