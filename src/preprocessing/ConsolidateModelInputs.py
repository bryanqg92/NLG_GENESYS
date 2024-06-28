from preprocessing.Text2PaddingSequences import Text2PaddingSequences
from preprocessing.GetFuzzySetNames import get_fuzzy_names
import numpy as np

class ConsolidateModelInputs:

    sequences = None
    pdlr_values = None

    @classmethod
    def GetConsolidateInputs(cls, pdlr_values:list) -> tuple:
        
        if not isinstance(pdlr_values, list):
            raise(TypeError("pdlr_values must be a list"))
        elif not all(isinstance(value, float) for value in pdlr_values):
            raise(ValueError("pdlr_values must contain only float values"))
        if len(pdlr_values) != 4:
            raise(ValueError("pdlr_values must contain 4 values"))
        
        cls.pdlr_values = np.array(pdlr_values).astype((float)).reshape(1,-1)
        cls.sequences = get_fuzzy_names(pdlr_values)
        cls.sequences = Text2PaddingSequences.PadFisSequences(cls.sequences)

        return cls.pdlr_values, cls.sequences

    