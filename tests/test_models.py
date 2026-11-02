import torch
from sed_portfolio.models import AudioCRNN, mlp_from_preset


def test_development_model_output_shape():
    model = mlp_from_preset("development").eval()
    with torch.no_grad():
        assert tuple(model(torch.randn(3, 32)).shape) == (3, 15)


def test_challenge_model_output_shape():
    model = mlp_from_preset("challenge").eval()
    with torch.no_grad():
        assert tuple(model(torch.randn(2, 960)).shape) == (2, 15)


def test_crnn_output_shape():
    model = AudioCRNN().eval()
    with torch.no_grad():
        assert tuple(model(torch.randn(2, 8, 128)).shape) == (2, 8, 15)
