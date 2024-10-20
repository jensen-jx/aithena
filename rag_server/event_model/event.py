from pydantic import BaseModel
from typing import Dict, Any, Optional
class Event(BaseModel):
    id: str
    text: str
    sources: Optional[Dict[str, str]]