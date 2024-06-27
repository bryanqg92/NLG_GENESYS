from preprocessing.Text2PaddingSequences import Text2PaddingSequences
from preprocessing.GetFuzzySetNames import get_fuzzy_names
import numpy as np

class ConsolidateModelInputs:

    sequences = None
    pdlr_values = None

    @classmethod
    def GetConsolidateInputs(cls, pdlr_values) -> tuple:
        cls.pdlr_values = np.array(pdlr_values).astype((float)).reshape(1,-1)
        cls.sequences = get_fuzzy_names(pdlr_values)
        cls.sequences = Text2PaddingSequences.PadFisSequences(cls.sequences)

        return cls.pdlr_values, cls.sequences

    