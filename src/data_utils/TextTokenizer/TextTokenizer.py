# data_utils/TextTokenizer.py

import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class TextTokenizer:

    tokenizer = None
    vocab_size = 0

    @classmethod
    def load_tokenizer(cls, tokenizer_path):
        """
        Carga el tokenizador desde un archivo JSON.

        Parámetros:
        tokenizer_path (str): La ruta al archivo JSON que contiene el tokenizador.
        """
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_json = f.read()
        cls.tokenizer = tokenizer_from_json(tokenizer_json)
        cls.vocab_size = len(cls.tokenizer.word_index) + 1

    @classmethod
    def tokenize_text(cls, text):
        """
        Tokeniza una cadena de texto.

        Parámetros:
        text (str): La cadena de texto a tokenizar.

        Retorna:
        Una lista de tokens.
        """
        if cls.tokenizer is None:
            raise ValueError("El tokenizador no ha sido cargado. Llame a load_tokenizer primero.")
        return cls.tokenizer.texts_to_sequences(text)

    @classmethod
    def detokenize_text(cls, tokens: np.ndarray) -> list:
        """
        Destokeniza una secuencia de tokens.

        Parámetros:
        tokens (list): Una lista de tokens.

        Retorna:
        La cadena de texto correspondiente a los tokens.
        """
        if cls.tokenizer is None:
            raise ValueError("El tokenizador no ha sido cargado. Llame a load_tokenizer primero.")
        
        return cls.tokenizer.sequences_to_texts(tokens)

    @classmethod
    def pad_sequences(cls, sequences, maxlen = None, dtype='int32', padding='post', truncating='post'):
        """
        Rellena las secuencias de tokens para que tengan la misma longitud.
k
        Parámetros:
        sequences (list): Una lista de secuencias de tokens.
        maxlen (int, opcional): La longitud máxima deseada de las secuencias. Si no se proporciona, se utilizará la longitud de la secuencia más larga.
        dtype (str, opcional): El tipo de datos que se utilizará para las secuencias rellenadas.
        padding (str, opcional): Tipo de relleno ('pre' o 'post').
        truncating (str, opcional): Tipo de truncamiento ('pre' o 'post').

        Retorna:
        Una matriz NumPy de secuencias rellenadas.
        """
        if maxlen is None:
            maxlen = cls.vocab_size
            
        return pad_sequences(sequences, maxlen=maxlen, dtype=dtype, padding=padding, truncating=truncating, value=0)
