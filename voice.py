import os
import asyncio
import json
import io
from websockets import serve
from google.cloud import speech_v1p1beta1 as speech
from pydub import AudioSegment

# Google Cloud Speech-to-Text 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "c:/Users/User/stone-climate-441309-e7-69ab160b40d4.json"  # 서비스 계정 키 파일 경로 설정

# WebSocket 연결 처리 함수
async def handle_connection(websocket, path):
    client = speech.SpeechClient()

    async for audio_data in websocket:
        try:
            print(f"Received audio data of length: {len(audio_data)}")

            # 오디오 데이터를 메모리 내에서 변환
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
            audio = audio.set_frame_rate(16000).set_channels(1)

            # 16-bit PCM 형식으로 메모리 내 오디오 파일 저장
            audio_bytes = io.BytesIO()
            audio.export(audio_bytes, format="wav", parameters=["-acodec", "pcm_s16le"])  # Ensure 16-bit PCM
            audio_bytes.seek(0)

            # Google Cloud Speech-to-Text 요청 생성
            audio_content = audio_bytes.read()
            audio = speech.RecognitionAudio(content=audio_content)
            diarization_config = speech.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                min_speaker_count=1,
                max_speaker_count=2
            )
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="ko-KR",
                diarization_config=diarization_config
            )

            response = client.recognize(config=config, audio=audio)

            # 인식된 결과 처리
            result = []
            if response.results:
                words_info = response.results[-1].alternatives[0].words
                current_speaker = None
                current_text = ""
                start_time = None

                for word_info in words_info:
                    # 화자 태그가 바뀌면 현재 문장 저장
                    if word_info.speaker_tag != current_speaker:
                        if current_speaker is not None and current_text:
                            result.append({
                                "start": start_time,
                                "end": word_info.start_time.total_seconds(),
                                "speaker": f"SPEAKER_{current_speaker}",
                                "text": current_text.strip()
                            })
                        # 새 화자 시작
                        current_speaker = word_info.speaker_tag
                        current_text = word_info.word
                        start_time = word_info.start_time.total_seconds()
                    else:
                        current_text += f" {word_info.word}"

                # 마지막 화자에 대한 정보 추가
                if current_text:
                    result.append({
                        "start": start_time,
                        "end": words_info[-1].end_time.total_seconds(),
                        "speaker": f"SPEAKER_{current_speaker}",
                        "text": current_text.strip()
                    })

            # 결과 전송
            await websocket.send(json.dumps(result))
            print(f"Sent transcription result: {result}")

        except Exception as e:
            print(f"Error processing audio data: {e}")
            await websocket.send(json.dumps({"error": str(e)}))

# WebSocket 서버 시작
async def main():
    async with serve(handle_connection, "localhost", 8765):
        print("WebSocket server started...")
        await asyncio.Future()  # 서버 계속 실행

# 서버 실행
asyncio.run(main())
