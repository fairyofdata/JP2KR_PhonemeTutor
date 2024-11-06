# 필요한 라이브러리와 모듈 불러오기
import pytube
import whisper
import torchaudio  # Wav2Vec 2.0 음성 분석
from langdetect import detect  # 텍스트의 언어 감지
from pydub import AudioSegment  # 오디오 데이터 전처리

# 1. 유튜브에서 데이터 수집
def download_audio_from_youtube(url):
    """유튜브 영상에서 오디오만 추출합니다."""
    yt = pytube.YouTube(url)
    stream = yt.streams.filter(only_audio=True).first()
    output_path = stream.download()
    audio = AudioSegment.from_file(output_path)
    return audio

# 2. Whisper를 이용한 음성 인식
def transcribe_audio_with_whisper(audio_path):
    """Whisper 모델을 사용하여 오디오를 텍스트로 변환합니다."""
    model = whisper.load_model("base")  # Whisper 모델 로드
    result = model.transcribe(audio_path)
    return result['text'], result['segments']  # 전체 텍스트 및 구간별 텍스트 반환

# 3. langdetect를 이용한 언어 구간 감지 및 분리
def detect_language_segments(transcribed_text):
    """각 구간의 언어를 감지하고, 한국어와 일본어 구간을 분리합니다."""
    segments = []
    current_lang = detect(transcribed_text[0])  # 첫 번째 구간의 언어 감지
    current_segment = []
    
    for text in transcribed_text:
        lang = detect(text)
        if lang == current_lang:
            current_segment.append(text)
        else:
            segments.append((current_lang, current_segment))  # 이전 언어 구간 저장
            current_lang = lang
            current_segment = [text]
    segments.append((current_lang, current_segment))  # 마지막 구간 추가
    return segments  # [(언어, 텍스트 구간 리스트)]

# 4. Wav2Vec 2.0을 이용한 실제 발음 분석
def analyze_pronunciation_with_wav2vec(audio_path, language_segments):
    """Wav2Vec 2.0을 통해 발음 분석을 수행합니다."""
    model, decoder, utils = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H  # Wav2Vec 2.0 모델 로드
    waveform, sample_rate = torchaudio.load(audio_path)
    
    results = []
    for lang, segment in language_segments:
        if lang == "ko":  # 한국어일 경우 발음 분석 수행
            for sentence in segment:
                # 음성 데이터와 텍스트 간의 차이를 비교하여 발음 오류를 탐지
                output = model(waveform)
                decoded_text = decoder(output[0])
                results.append((sentence, decoded_text))
    return results  # [(원래 텍스트, 실제 발음 분석 텍스트)]

# 5. 발음 오류 교정 피드백 제공
def provide_feedback(original_text, analyzed_text):
    """원래 텍스트와 분석된 발음을 비교하여 교정 피드백을 생성합니다."""
    feedback = []
    for original, analyzed in zip(original_text, analyzed_text):
        if original != analyzed:
            feedback.append(f"발음 오류 발견: '{original}' 대신 '{analyzed}'로 발음되었습니다.")
        else:
            feedback.append("올바르게 발음되었습니다.")
    return feedback

# 메인 실행 흐름
def main():
    # 예시 URL
    url = "https://www.youtube.com/watch?v=example"
    
    # 1. 유튜브에서 오디오 추출
    audio = download_audio_from_youtube(url)
    audio_path = "downloaded_audio.wav"
    audio.export(audio_path, format="wav")
    
    # 2. Whisper로 음성을 텍스트로 변환
    transcribed_text, segments = transcribe_audio_with_whisper(audio_path)
    
    # 3. 언어 구간 감지 및 분리
    language_segments = detect_language_segments(transcribed_text)
    
    # 4. Wav2Vec 2.0을 이용한 발음 분석
    pronunciation_results = analyze_pronunciation_with_wav2vec(audio_path, language_segments)
    
    # 5. 피드백 제공
    feedback = provide_feedback(transcribed_text, [result[1] for result in pronunciation_results])
    
    # 결과 출력
    for fb in feedback:
        print(fb)

# 실행
if __name__ == "__main__":
    main()
