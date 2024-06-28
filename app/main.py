from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from src.predict.predictor import Predictor
from typing import List

app = FastAPI()

MODEL_PATH = "src/models/seq2seqLSTM_model.pt"
TOKENIZER_PATH = "src/data_utils/TextTokenizer/tokenizer.json"

# Inicializar el modelo al arrancar la aplicación
@app.on_event("startup")
def load_model():
    global predictor
    predictor = Predictor(MODEL_PATH, TOKENIZER_PATH)

class PredictionResponse(BaseModel):
    predictions: List[str]

@app.post("/predict", response_model=PredictionResponse, summary="Realizar predicciones a partir de un archivo de texto")
async def predict(file: UploadFile = File(..., description="Archivo de texto con las entradas separadas por líneas")):
    """
    Realiza predicciones a partir de un archivo de texto.

    - **file**: Archivo de texto con las entradas separadas por líneas. Cada línea debe contener 4 valores separados por " / ".
    """
    if file.content_type != "text/plain":
        raise HTTPException(status_code=400, detail="Only accept (.txt) files")

    contents = await file.read()
    text = contents.decode('utf-8')
    predictions = []

    for i, line in enumerate(text.split("\n")):
        if not line.strip():
            continue  # Ignora las líneas vacías
        values = line.split(" / ")
        if len(values) != 4:
            raise HTTPException(status_code=400, detail=f"Error in line {i+1}. Must contain exactly 4 values")
        try:
            values = [float(value) if not value.isdigit() else int(value) for value in values]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Error converting values in line {i+1}: {str(e)}")
        
        # predictor.predict debe devolver un str, no una lista con un solo elemento
        prediction = predictor.predict(values)[0]  # Asegúrate de obtener el primer elemento de la lista devuelta
        predictions.append(prediction)

    return PredictionResponse(predictions=predictions)

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
