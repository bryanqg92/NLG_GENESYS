import pytest
from preprocessing.ConsolidateModelInputs import ConsolidateModelInputs
from preprocessing.Text2PaddingSequences import Text2PaddingSequences
from preprocessing.GetFuzzySetNames import get_fuzzy_names
from data_utils.TextTokenizer.TextTokenizer import TextTokenizer
import numpy as np

class TestConsolidateModelInputs:
    
    @staticmethod
    @pytest.fixture(scope="module", autouse=True)
    def setup_tokenizer():
        tokenizer_path = 'src/data_utils/TextTokenizer/tokenizer.json'
        TextTokenizer.load_tokenizer(tokenizer_path)
    
    def test_valid_input(self):
        pdlr_values = [1.0, 2.0, 3.0, 4.0]  # Example valid input
        pdlr_values_expected = np.array(pdlr_values).astype(float).reshape(1, -1)
        sequences_expected = get_fuzzy_names(pdlr_values)
        sequences_expected = Text2PaddingSequences.PadFisSequences(sequences_expected)
        
        pdlr_values_output, sequences_output = ConsolidateModelInputs.GetConsolidateInputs(pdlr_values)
        
        assert np.array_equal(pdlr_values_output, pdlr_values_expected), "PDLR values do not match"
        assert np.array_equal(sequences_output, sequences_expected), "Sequences do not match"
    
    def test_invalid_length(self):
        pdlr_values = [1.0, 2.0, 3.0]  # Example input with invalid length
        with pytest.raises(ValueError, match="pdlr_values must contain 4 values"):
            ConsolidateModelInputs.GetConsolidateInputs(pdlr_values)
    
    def test_invalid_type(self):
        pdlr_values = 1234  # Example input of invalid type (integer instead of list)
        with pytest.raises(TypeError, match="pdlr_values must be a list"):
            ConsolidateModelInputs.GetConsolidateInputs(pdlr_values)
    
    def test_non_numeric_values(self):
        pdlr_values = ["1.0", "2.0", "3.0", "4.0"]  # Example input with non-numeric strings
        with pytest.raises(ValueError, match="pdlr_values must contain only float values"):
            ConsolidateModelInputs.GetConsolidateInputs(pdlr_values)
    
    def test_mix_numeric_strings(self):
        pdlr_values = ["1.0", 2.0, "3.0", 4.0]  # Example input with mixed numeric strings and floats
        with pytest.raises(ValueError, match="pdlr_values must contain only float values"):
            ConsolidateModelInputs.GetConsolidateInputs(pdlr_values)
