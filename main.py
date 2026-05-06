from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import json
from model import model

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    smiles: str = Form(""),
    blend: str = Form("")
):
    parsed_blend = None

    if blend:
        try:
            parsed_blend = json.loads(blend)
        except:
            parsed_blend = None

    result = model.get_properties_from_mixture(
        smiles=smiles if smiles else None,
        blend=parsed_blend
    )

    return templates.TemplateResponse("index.html", {
        "request": request,
        "forward_result": result
    })


@app.post("/inverse", response_class=HTMLResponse)
def inverse(
    request: Request,
    octane: float = Form(...),
    cetane: float = Form(...),
    flash_point: float = Form(...)
):
    result = model.get_mixture_from_properties(
        octane, cetane, flash_point
    )

    return templates.TemplateResponse("index.html", {
        "request": request,
        "inverse_result": result
    })