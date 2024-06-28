from src.predict.predictor import Predictor

class load_nlg:

    predictor = None
    
    @classmethod
    def load_nlg_model(cls):
        cls.predictor = Predictor(
            model_path=MODEL_PATH,
            tokenizer_path=TOKENIZER_PATH
        )