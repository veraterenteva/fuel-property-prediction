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

@app.post(
    "/predict",
    response_class=HTMLResponse
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

@app.post(
    "/inverse",
    response_class=HTMLResponse
)

def inverse(
    request:Request,
    ron:float=Form(...),
    mon:float=Form(...),
    k:int=Form(4)
):
    result=(model.get_mixture_from_properties(ron, mon, k))

    return templates.TemplateResponse(
        "index.html",
        {
            "request":
            request,
            "inverse_result":
            result
        }
    )