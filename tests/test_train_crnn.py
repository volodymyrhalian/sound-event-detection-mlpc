import torch
from sed_portfolio.train_crnn import collate_sequences


def test_padded_batch_mask():
    batch = [(torch.ones(3, 128), torch.ones(3, 15)),
             (torch.ones(2, 128), torch.zeros(2, 15))]
    x, y, mask = collate_sequences(batch)
    assert x.shape == (2, 3, 128)
    assert y.shape == (2, 3, 15)
    assert mask.tolist() == [[True, True, True], [True, True, False]]
