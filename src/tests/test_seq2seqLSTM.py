import pytest
import torch
import torch.nn as nn
from models.seq2seq_model import seq2seqLSTM

@pytest.fixture
def model():
    input_size = 10
    hidden_size = 20
    num_decoder_tokens = 30
    num_layers = 1
    model = seq2seqLSTM(input_size, hidden_size, num_decoder_tokens, num_layers)
    return model

def test_seq2seqLSTM_forward_shape(model):
    batch_size = 5
    seq_len = 7
    input_size = 10
    num_decoder_tokens = 30

    inputs = torch.randn(batch_size, seq_len, input_size)
    dec_inputs = torch.randint(0, num_decoder_tokens, (batch_size, seq_len))

    outputs = model(inputs, dec_inputs)

    assert outputs.shape == (batch_size, seq_len, num_decoder_tokens)

def test_seq2seqLSTM_forward_values(model):
    batch_size = 5
    seq_len = 7
    input_size = 10
    num_decoder_tokens = 30

    inputs = torch.randn(batch_size, seq_len, input_size)
    dec_inputs = torch.randint(0, num_decoder_tokens, (batch_size, seq_len))

    outputs = model(inputs, dec_inputs)

    assert not torch.isnan(outputs).any()
    assert torch.isfinite(outputs).all()

def test_seq2seqLSTM_zero_input(model):
    batch_size = 5
    seq_len = 7
    input_size = 10
    num_decoder_tokens = 30

    inputs = torch.zeros(batch_size, seq_len, input_size)
    dec_inputs = torch.randint(0, num_decoder_tokens, (batch_size, seq_len))

    outputs = model(inputs, dec_inputs)

    assert outputs.shape == (batch_size, seq_len, num_decoder_tokens)
    assert not torch.isnan(outputs).any()
    assert torch.isfinite(outputs).all()

def test_seq2seqLSTM_random_input(model):
    batch_size = 5
    seq_len = 7
    input_size = 10
    num_decoder_tokens = 30

    inputs = torch.randn(batch_size, seq_len, input_size)
    dec_inputs = torch.randint(0, num_decoder_tokens, (batch_size, seq_len))

    outputs = model(inputs, dec_inputs)

    assert outputs.shape == (batch_size, seq_len, num_decoder_tokens)
    assert not torch.isnan(outputs).any()
    assert torch.isfinite(outputs).all()

def test_seq2seqLSTM_different_batch_sizes(model):
    seq_len = 7
    input_size = 10
    num_decoder_tokens = 30

    for batch_size in [1, 10, 50]:
        inputs = torch.randn(batch_size, seq_len, input_size)
        dec_inputs = torch.randint(0, num_decoder_tokens, (batch_size, seq_len))

        outputs = model(inputs, dec_inputs)

        assert outputs.shape == (batch_size, seq_len, num_decoder_tokens)
        assert not torch.isnan(outputs).any()
        assert torch.isfinite(outputs).all()

def test_seq2seqLSTM_different_sequence_lengths(model):
    batch_size = 5
    input_size = 10
    num_decoder_tokens = 30

    for seq_len in [1, 15, 30]:
        inputs = torch.randn(batch_size, seq_len, input_size)
        dec_inputs = torch.randint(0, num_decoder_tokens, (batch_size, seq_len))

        outputs = model(inputs, dec_inputs)

        assert outputs.shape == (batch_size, seq_len, num_decoder_tokens)
        assert not torch.isnan(outputs).any()
        assert torch.isfinite(outputs).all()

def test_seq2seqLSTM_different_input_sizes(model):
    batch_size = 5
    seq_len = 7
    num_decoder_tokens = 30

    for input_size in [5, 20, 50]:
        model = seq2seqLSTM(input_size, 20, num_decoder_tokens)
        inputs = torch.randn(batch_size, seq_len, input_size)
        dec_inputs = torch.randint(0, num_decoder_tokens, (batch_size, seq_len))

        outputs = model(inputs, dec_inputs)

        assert outputs.shape == (batch_size, seq_len, num_decoder_tokens)
        assert not torch.isnan(outputs).any()
        assert torch.isfinite(outputs).all()

def test_seq2seqLSTM_different_hidden_sizes(model):
    batch_size = 5
    seq_len = 7
    input_size = 10
    num_decoder_tokens = 30

    for hidden_size in [10, 40, 100]:
        model = seq2seqLSTM(input_size, hidden_size, num_decoder_tokens)
        inputs = torch.randn(batch_size, seq_len, input_size)
        dec_inputs = torch.randint(0, num_decoder_tokens, (batch_size, seq_len))

        outputs = model(inputs, dec_inputs)

        assert outputs.shape == (batch_size, seq_len, num_decoder_tokens)
        assert not torch.isnan(outputs).any()
        assert torch.isfinite(outputs).all()



