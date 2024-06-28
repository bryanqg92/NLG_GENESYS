from src.data_utils.TextTokenizer.TextTokenizer import TextTokenizer
import numpy as np

class Text2PaddingSequences:
    """
    Clase para convertir una lista de textos en secuencias de tokens con relleno (padding).

    Métodos:
        PadFisSequences(FisText: list) -> np.ndarray:
            Convierte una lista de textos en secuencias de tokens con relleno.

            Argumentos:
                FisText (list): Lista de textos a convertir.

            Retorna:
                np.ndarray: Arreglo de NumPy con las secuencias de tokens con relleno.
    """

    @classmethod
    def PadFisSequences(cls, FisText: list) -> np.ndarray:
        """
        Convierte una lista de textos en secuencias de tokens con relleno.

        Argumentos:
            FisText (list): Lista de textos a convertir.

        Retorna:
            np.ndarray: Arreglo de NumPy con las secuencias de tokens con relleno.
        """
        decoder_inputs = TextTokenizer.tokenize_text(FisText)
        decoder_inputs = cls._Padsequences(decoder_inputs)
        return decoder_inputs

    @classmethod
    def _Padsequences(cls, decoder_inputs: list) -> np.ndarray:
        """
        Aplica relleno (padding) a una lista de secuencias de tokens.

        Argumentos:
            decoder_inputs (list): Lista de secuencias de tokens.

        Retorna:
            np.ndarray: Arreglo de NumPy con las secuencias de tokens con relleno.
        """
        return TextTokenizer.pad_sequences(decoder_inputs)



