from fastapi import FastAPI
from app.routers.predict import nlg_model_r
from app.dependencies.load_nlg_model import load_nlg
import logging 

app = FastAPI()
app.include_router(nlg_model_r)

MODEL_PATH = "src/models/seq2seqLSTM_model.pt"
TOKENIZER_PATH = "src/data_utils/TextTokenizer/tokenizer.json"

@app.on_event("startup")
def load_model():
    load_nlg.load_nlg_model(MODEL_PATH,TOKENIZER_PATH)
    logging.info("Modelo cargado")


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
