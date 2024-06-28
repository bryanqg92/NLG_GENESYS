from src.predict.predictor import Predictor

class load_nlg:

    predictor = None
    
    @classmethod
    def load_nlg_model(cls, model_path:str, tokenizer_path:str):
        cls.predictor = Predictor(model_path,tokenizer_path)