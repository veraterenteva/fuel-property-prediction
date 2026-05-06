from typing import List, Optional
from pydantic import BaseModel


class BlendComponent(BaseModel):
    smiles: str
    fraction: float


class ForwardRequest(BaseModel):
    smiles: Optional[str] = None
    blend: Optional[List[BlendComponent]] = None


class InverseRequest(BaseModel):
    octane: float
    cetane: float
    flash_point: float