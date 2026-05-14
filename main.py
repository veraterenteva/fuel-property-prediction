from typing import Optional

from fastapi import FastAPI
from fastapi import Request
from fastapi import Form

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import json

from fuel_model import model

app=FastAPI()

templates=Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)

def home(request:Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request":request
        }
    )

@app.post("/predict")
def predict(request: Request, blend: str = Form(...)):
    try:
        parsed_blend = json.loads(blend)
        result = model.get_properties_from_mixture(parsed_blend)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "forward_result": result
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "forward_result": {
                    "error": str(e)
                }
            }
        )

@app.post("/inverse", response_class=HTMLResponse)
def inverse(
    request: Request,
    ron: Optional[float] = Form(None),
    mon: Optional[float] = Form(None),
    cn: Optional[float] = Form(None),
    k: int = Form(4)
):

    targets = {
        "ron": ron,
        "mon": mon,
        "cn": cn
    }

    # Хотя бы одно значение должно быть задано
    if all(v is None for v in targets.values()):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "inverse_result": {
                    "error": "At least one of RON, MON, CN must be provided"
                }
            }
        )

    # Диапазоны величин
    def validate(name, value, min_v, max_v):
        if value is None:
            return None
        if not (min_v <= value <= max_v):
            raise ValueError(
                f"{name} must be in range [{min_v}, {max_v}]"
            )
        return value

    try:
        ron = validate("RON", ron, 0, 120)
        mon = validate("MON", mon, 0, 120)
        cn  = validate("CN", cn, 0, 100)

    except ValueError as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "inverse_result": {
                    "error": str(e)
                }
            }
        )

    result = model.get_mixture_from_properties(
        ron,
        mon,
        cn,
        k
    )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "inverse_result": result
        }
    )