from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import uuid, os

from pipelines.vision import caption_image
from pipelines.asr import transcribe_audio
from pipelines.reasoner import explain_world

app = FastAPI(title="ExplainMyWorld")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # allow_origins=["http://localhost:3000"]
)

TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

class ExplainResponse(BaseModel):
    observed: list[str]
    likely_causes: list[str]
    why: str
    confidence: str
    question: str
    image_caption: str
    transcript: str | None = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=ExplainResponse)
async def analyze(
    image: UploadFile = File(...),
    audio: UploadFile | None = File(None),
    text: str | None = Form(None),
):
    # 1) Image caption
    img = Image.open(image.file).convert("RGB")
    img_caption = caption_image(img)

    # 2) Audio -> transcript (optional)
    transcript = None
    if audio is not None:
        audio_id = str(uuid.uuid4())
        audio_path = os.path.join(TMP_DIR, f"{audio_id}_{audio.filename}")
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        transcript = transcribe_audio(audio_path)

    # 3) Merge text + transcript
    user_context = " ".join(x for x in [text, transcript] if x and x.strip())

    # 4) Reasoning
    result = explain_world(img_caption, user_context)

    return ExplainResponse(
        observed=result.get("observed", []) or [],
        likely_causes=result.get("likely_causes", []) or [],
        why=result.get("why", "") or "",
        confidence=result.get("confidence", "low") or "low",
        question=result.get("question", "What time of day and where was this taken?") or "",
        image_caption=img_caption,
        transcript=transcript,
    )
