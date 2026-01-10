# -*- coding: utf-8 -*-
'''
@File    :   difyASR.py
@Author  :   一力辉
'''


from ..builder import ASREngines
from ..engineBase import BaseASREngine
import io, base64, json
from digitalHuman.protocol import AudioMessage, TextMessage, AUDIO_TYPE
from digitalHuman.utils import logger, httpxAsyncClient, wavToMp3

__all__ = ["DifyApiAsr"]


@ASREngines.register("Dify")
class DifyApiAsr(BaseASREngine): 
    async def run(self, input: AudioMessage, **kwargs) -> TextMessage:
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
        message = TextMessage(data=result)
        return message