from pydantic import BaseModel
from datetime import datetime

class DocumentMetadata(BaseModel):
    filename: str
    file_type: str
    page_count: int | None
    word_count: int
    upload_timestamp: datetime
    file_size_kb: float