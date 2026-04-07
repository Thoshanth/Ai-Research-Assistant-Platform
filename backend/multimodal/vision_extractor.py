import os
import base64
from groq import Groq
from dotenv import load_dotenv
from backend.logger import get_logger

load_dotenv()
logger = get_logger("multimodal.vision")

_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Groq's free vision model
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def image_to_base64(image_path: str) -> str:
    """
    Reads an image file and converts it to base64 string.
    Vision LLMs require images in base64 format embedded
    directly in the API request.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_bytes_to_base64(image_bytes: bytes) -> str:
    """
    Converts raw image bytes to base64.
    Used when we render PDF pages as images in memory.
    """
    return base64.b64encode(image_bytes).decode("utf-8")


def extract_text_from_image(
    image_base64: str,
    media_type: str = "image/png",
    instruction: str = None,
) -> str:
    """
    Sends an image to the vision LLM and extracts text from it.

    This is the core multimodal function. The LLM receives:
    - The image encoded as base64
    - A text instruction telling it what to do

    It returns all text it can see in the image, preserving
    structure like tables, bullet points, and headers.
    """
    if instruction is None:
        instruction = """Extract ALL text from this image exactly as it appears.
Preserve the structure including:
- Headers and titles
- Bullet points and numbered lists  
- Table contents (row by row)
- Any handwritten text you can read
- Numbers, dates, and special characters

Return only the extracted text, nothing else."""

    logger.info(f"Vision LLM request | model={VISION_MODEL} | media_type={media_type}")

    response = _groq.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_base64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": instruction,
                    },
                ],
            }
        ],
        max_tokens=4096,
        temperature=0.1,
    )

    extracted = response.choices[0].message.content
    logger.info(f"Vision extraction complete | chars={len(extracted)}")
    return extracted


def describe_image(image_base64: str, media_type: str = "image/jpeg") -> str:
    """
    Generates a detailed description of what's in an image.
    Used when the user uploads a photo rather than a document.

    Different from extract_text — this understands the full
    visual content including objects, charts, diagrams, scenes.
    """
    logger.info("Generating image description")

    instruction = """Provide a detailed description of this image including:
1. What type of image is this (photo, diagram, chart, screenshot, etc.)
2. Main content and subjects visible
3. Any text visible in the image
4. If it's a chart/graph: describe the data it shows
5. If it's a document: describe its structure and key information
6. Any other relevant visual details

Be thorough and specific."""

    response = _groq.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_base64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": instruction,
                    },
                ],
            }
        ],
        max_tokens=1024,
        temperature=0.3,
    )

    description = response.choices[0].message.content
    logger.info(f"Image description complete | chars={len(description)}")
    return description