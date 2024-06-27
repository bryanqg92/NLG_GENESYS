import torch
import numpy as np
from models.seq2seq_model import seq2seqLSTM
from data_utils.TextTokenizer.TextTokenizer import *
from preprocessing.ConsolidateModelInputs import ConsolidateModelInputs

class Predictor:
    """
    Clase para realizar predicciones utilizando un modelo seq2seq previamente entrenado.

    Args:
        model_path (str): Ruta del archivo que contiene los pesos del modelo entrenado.
        tokenizer_path (str): Ruta del archivo que contiene el tokenizador utilizado durante el entrenamiento.
    """

    def __init__(self, model_path: str, tokenizer_path: str):
        """
        Inicializa el objeto Predictor.

        Args:
            model_path (str): Ruta del archivo que contiene los pesos del modelo entrenado.
            tokenizer_path (str): Ruta del archivo que contiene el tokenizador utilizado durante el entrenamiento.
        """
        # Cargar el tokenizador utilizado durante el entrenamiento
        TextTokenizer.load_tokenizer(tokenizer_path)
        # Configurar el dispositivo (CPU o GPU) para realizar las predicciones
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cargar el modelo entrenado y moverlo al dispositivo configurado
        self.model = seq2seqLSTM(4, 64, TextTokenizer.vocab_size).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()



    def predict(self, input_list: list) -> list:
        """
        Realiza predicciones utilizando el modelo entrenado.

        Args:
            input_list (list): Lista de entradas para realizar las predicciones.

        Returns:
            list: Lista de predicciones en formato de texto.
        """
        encoder_inputs, decoder_inputs = self._preprocess_inputs(input_list)

        with torch.no_grad():
            preds = self.model(encoder_inputs, decoder_inputs)
            _, preds_max = torch.max(preds.data, 1)
            text_preds = self._detokenize_predictions(preds_max.cpu().numpy())

        return text_preds

    def _preprocess_inputs(self, input_list: list) -> tuple:
        """
        Preprocesa las entradas para el modelo.

        Args:
            input_list (list): Lista de entradas para realizar las predicciones.

        Returns:
            tuple: Tupla que contiene las entradas preprocesadas para el encoder y el decoder.
        """
        encoder_inputs, decoder_inputs = ConsolidateModelInputs.GetConsolidateInputs(input_list)
        encoder_inputs = torch.tensor(encoder_inputs, dtype=torch.float32).view(1, 1, -1).to(self.device)
        decoder_inputs = torch.tensor(decoder_inputs, dtype=torch.int).to(self.device)
        return encoder_inputs, decoder_inputs

    @staticmethod
    def _detokenize_predictions(preds: np.ndarray) -> list:
        """
        Convierte las predicciones tokenizadas en texto legible.

        Args:
            preds (np.ndarray): Arreglo de predicciones tokenizadas.

        Returns:
            list: Lista de predicciones en formato de texto.
        """
        preds_words = []
        for pred_row in preds:
            pred_row_words = []
            for idx in pred_row:
                if idx in TextTokenizer.tokenizer.index_word:
                    pred_row_words.append(TextTokenizer.tokenizer.index_word[idx])
            preds_words.append(' '.join(pred_row_words))
        return preds_words
