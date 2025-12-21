from faster_whisper import WhisperModel

_model = WhisperModel("small", device="cpu", compute_type="int8")

def transcribe_audio(audio_path: str) -> str:
    segments, _info = _model.transcribe(audio_path, beam_size=5)
    return " ".join(seg.text.strip() for seg in segments).strip()
