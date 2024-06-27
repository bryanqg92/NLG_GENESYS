import pytest
from data_utils.TextTokenizer.TextTokenizer import TextTokenizer
import numpy as np
import json
import os

# Datos de prueba
sample_text = ["Hola mundo", "Prueba de tokenización", "Pytest es genial"]
sample_tokens = [[1, 2], [3, 4, 5], [6, 7, 8]]


tokenizer_json = {
    "class_name": "Tokenizer",
    "config": {
        "num_words": None,
        "filters": '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        "lower": True,
        "split": " ",
        "char_level": False,
        "oov_token": None,
        "document_count": 0,   
        "word_counts": json.dumps({"hola": 1, "mundo": 1, "prueba": 1, "de": 1, "tokenización": 1, "pytest": 1, "es": 1, "genial": 1}),
        "word_docs": json.dumps({"hola": 1, "mundo": 1, "prueba": 1, "de": 1, "tokenización": 1, "pytest": 1, "es": 1, "genial": 1}),
        "index_docs": json.dumps({"1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 1}),
        "index_word": json.dumps({"1": "hola", "2": "mundo", "3": "prueba", "4": "de", "5": "tokenización", "6": "pytest", "7": "es", "8": "genial"}),
        "word_index": json.dumps({"hola": 1, "mundo": 2, "prueba": 3, "de": 4, "tokenización": 5, "pytest": 6, "es": 7, "genial": 8})
}
}


tokenizer_path = 'test_tokenizer.json'
with open(tokenizer_path, 'w', encoding='utf-8') as f:
    json.dump(tokenizer_json, f)

@pytest.fixture(scope="module", autouse=True)
def setup_tokenizer():

    TextTokenizer.load_tokenizer(tokenizer_path)
    yield
    # Limpia después de las pruebas
    os.remove(tokenizer_path)

def test_load_tokenizer():
    assert TextTokenizer.tokenizer is not None
    assert TextTokenizer.vocab_size == 9  # 8 palabras + 1 para padding

def test_tokenize_text():
    tokens = TextTokenizer.tokenize_text(sample_text)
    assert len(tokens) == len(sample_text)
    assert tokens == [[1, 2], [3, 4, 5], [6, 7, 8]]

def test_detokenize_text():
    detokenized_text = TextTokenizer.detokenize_text(sample_tokens)
    assert len(detokenized_text) == len(sample_tokens)
    assert detokenized_text == ["hola mundo", "prueba de tokenización", "pytest es genial"]

def test_pad_sequences():
    padded_sequences = TextTokenizer.pad_sequences(sample_tokens, maxlen=5)
    assert padded_sequences.shape == (3, 5)
    assert np.array_equal(padded_sequences, np.array([
        [1, 2, 0, 0, 0],
        [3, 4, 5, 0, 0],
        [6, 7, 8, 0, 0]
    ]))

def test_pad_sequences_no_maxlen():
    padded_sequences = TextTokenizer.pad_sequences(sample_tokens)
    assert padded_sequences.shape == (3, TextTokenizer.vocab_size)
    assert np.array_equal(padded_sequences, np.array([
        [1, 2, 0, 0, 0, 0, 0, 0, 0],
        [3, 4, 5, 0, 0, 0, 0, 0, 0],
        [6, 7, 8, 0, 0, 0, 0, 0, 0]
    ]))

