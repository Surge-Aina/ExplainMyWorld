from transformers import pipeline
from PIL import Image

_captioner = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-large",
)

def caption_image(image: Image.Image) -> str:
    out = _captioner(image)
    return out[0]["generated_text"].strip()
