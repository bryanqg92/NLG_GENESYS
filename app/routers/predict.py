from app.dependencies.load_nlg_model import load_nlg
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel
from typing import List
import datetime
import os

nlg_model_r = APIRouter()

class FileName(BaseModel):
    file_name: str

class SinglePredictionResponse(BaseModel):
    prediction: str

class PredictionRequest(BaseModel):
    values: List[float]

@nlg_model_r.post("/predict_file", response_model=FileName, summary="Realizar predicciones a partir de un archivo de texto.")
async def file_predict(file: UploadFile = File(..., description="Archivo de texto con las entradas separadas por líneas")):
    """
    Realiza predicciones a partir de un archivo de texto.

    - **file**: Archivo de texto con las entradas separadas por líneas. Cada línea debe contener 4 valores separados por " / ".
    """
    if file.content_type != "text/plain":
        raise HTTPException(status_code=400, detail="Only accept (.txt) files")

    contents = await file.read()
    text = contents.decode('utf-8')
    predictions = []

    file_name = "GeNeBot" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = f'app/temp/'

    with open(f'{file_path}{file_name}.txt', "w") as f:
        for i, line in enumerate(text.split("\n")):
            if not line.strip():
                continue  
            values = line.split(" / ")
            if len(values) != 4:
                os.remove(f'{file_path}{file_name}.txt')
                raise HTTPException(status_code=400, detail=f"Error in line {i+1}. Must contain exactly 4 values")

            try:
                values = [float(value) if not value.isdigit() else int(value) for value in values]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Error converting values in line {i+1}: {str(e)}")

            prediction = load_nlg.predictor.predict(values)[0] 
            predictions.append(prediction)
            f.write(prediction + "\n")

    return FileName(file_name=f'{file_name}.txt')


@nlg_model_r.post("/predict", response_model=SinglePredictionResponse, summary="Realizar predicciones a partir de una lista de valores PDLR.")
async def predict(request: PredictionRequest):
    values = request.values

    if not isinstance(values, list) or len(values) != 4:
        raise HTTPException(status_code=400, detail="Los valores ingresados no son una lista o no son 4 valores")

    prediction = load_nlg.predictor.predict(values)[0]
    return SinglePredictionResponse(prediction=prediction)

@nlg_model_r.get("/download_pred_file/{file_name}", summary="Descargar archivo de predicciones")
async def download_pred_file(file_name: str):
    file_path = f'app/temp/{file_name}'

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    def remove_file(path: str):
        os.remove(path)

    return FileResponse(
        file_path, 
        filename=f"{file_name}",
        background=BackgroundTask(remove_file, file_path)
    )