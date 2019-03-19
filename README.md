# Torch CRF
[![CircleCI](https://circleci.com/gh/s14t284/TorchCRF.svg?style=svg)](https://circleci.com/gh/s14t284/TorchCRF) [![Coverage Status](https://coveralls.io/repos/github/s14t284/TorchCRF/badge.svg)](https://coveralls.io/github/s14t284/TorchCRF)

Implementation of CRF (Conditional Random Fields) in PyTorch 1.0

## Requirements

- python3 (>=3.6)
- PyTorch 1.0

## Installation

    $ pip install TorchCRF

## Usage

```python
>>> import torch
>>> from TorchCRF import CRF
>>> batch_size = 2
>>> sequence_size = 3
>>> num_labels = 5
>>> mask = torch.FloatTensor([[1, 1, 1], [1, 1, 0]]) # (batch_size. sequence_size)
>>> labels = torch.LongTensor([[0, 2, 3], [1, 4, 1]])  # (batch_size, sequence_size)
>>> hidden = torch.randn((batch_size, sequence_size, num_labels), requires_grad=True)
>>> crf = CRF(num_labels)
```

### Computing log-likelihood (used where forward)

```python
>>> crf.forward(hidden, labels, mask)
tensor([-7.6204, -3.6124], grad_fn=<ThSubBackward>)
```

### Decoding (predict labels of sequences)

```python
>>> crf.viterbi_decode(hidden, mask)
[[0, 2, 2], [4, 0]]
```

## License

MIT

## References

- [threelittlemonkeys/lstm-crf-pytorch](https://github.com/threelittlemonkeys/lstm-crf-pytorch)
- [kmkurn/pytorch-crf](https://github.com/kmkurn/pytorch-crf)
