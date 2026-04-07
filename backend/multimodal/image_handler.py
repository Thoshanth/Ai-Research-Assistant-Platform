import os
from pathlib import Path
from backend.multimodal.vision_extractor import (
    image_to_base64,
    extract_text_from_image,
    describe_image,
)
from backend.logger import get_logger

logger = get_logger("multimodal.image_handler")

SUPPORTED_IMAGE_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}


def process_image_file(file_path: str) -> dict:
    """
    Processes a direct image upload.

    Does two things:
    1. Extracts any text visible in the image (OCR via vision LLM)
    2. Generates a description of the image content

    Returns both so they can be stored and searched separately.
    """
    extension = Path(file_path).suffix.lower().lstrip(".")
    media_type = SUPPORTED_IMAGE_TYPES.get(extension, "image/jpeg")

    logger.info(f"Processing image | file='{file_path}' | type={media_type}")

    # Convert image to base64
    image_base64 = image_to_base64(file_path)

    # Extract text from image
    logger.info("Extracting text from image via vision LLM")
    extracted_text = extract_text_from_image(
        image_base64=image_base64,
        media_type=media_type,
    )

    # Generate description
    logger.info("Generating image description via vision LLM")
    description = describe_image(
        image_base64=image_base64,
        media_type=media_type,
    )

    # Combine text and description for storage
    # This makes the image fully searchable via RAG
    combined_text = f"""=== IMAGE DESCRIPTION ===
{description}

=== EXTRACTED TEXT ===
{extracted_text}"""

    file_size_kb = round(os.path.getsize(file_path) / 1024, 2)

    logger.info(
        f"Image processing complete | "
        f"text_chars={len(extracted_text)} | "
        f"desc_chars={len(description)}"
    )

    return {
        "extracted_text": combined_text,
        "description": description,
        "text_only": extracted_text,
        "file_size_kb": file_size_kb,
        "media_type": media_type,
    }