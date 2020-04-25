import unittest
import torch
from TorchCRF import CRF


class TestCRF(unittest.TestCase):
    def setUp(self):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 2
        self.sequence_size = 3
        self.num_labels = 5
        self.crf = CRF(self.num_labels)
        self.mask = torch.ByteTensor([[1, 1, 1], [1, 1, 0]]).to(device)
        self.labels = torch.LongTensor([[0, 2, 3], [1, 4, 1]]).to(device)
        self.hidden = torch.autograd.Variable(
            torch.randn(self.batch_size, self.sequence_size, self.num_labels),
            requires_grad=True,
        ).to(device)

    def test_initialize_variables(self):

        self.assertEqual(self.crf.num_labels, self.num_labels)
        self.assertEqual(
            self.crf.trans_matrix.size(), (self.num_labels, self.num_labels)
        )
        self.assertEqual(self.crf.start_trans.size(), (self.num_labels,))
        self.assertEqual(self.crf.end_trans.size(), (self.num_labels,))

        num_labels = -1
        with self.assertRaises(ValueError) as er:
            CRF(num_labels)
        exception = er.exception
        self.assertEqual(exception.args[0], "invalid number of labels: -1")

    def test_initialize_score(self):

        self.assertTrue(0.1 > torch.max(self.crf.trans_matrix))
        self.assertTrue(-0.1 < torch.min(self.crf.trans_matrix))
        self.assertTrue(0.1 > torch.max(self.crf.start_trans))
        self.assertTrue(-0.1 < torch.min(self.crf.start_trans))
        self.assertTrue(0.1 > torch.max(self.crf.end_trans))
        self.assertTrue(-0.1 < torch.min(self.crf.end_trans))

    def test_forward(self):
        fvalue = self.crf.forward(self.hidden, self.labels, self.mask)
        if torch.cuda.is_available():
            self.assertEqual(fvalue.type(), "torch.cuda.FloatTensor")
        else:
            self.assertEqual(fvalue.type(), "torch.FloatTensor")
        self.assertEqual(fvalue.size(), (self.batch_size,))

    def test_compute_log_likelihood(self):
        # log likelihood of denominator term
        dllh = self.crf._compute_denominator_log_likelihood(self.hidden, self.mask)
        if torch.cuda.is_available():
            self.assertEqual(dllh.type(), "torch.cuda.FloatTensor")
        else:
            self.assertEqual(dllh.type(), "torch.FloatTensor")
        self.assertEqual(dllh.size(), (self.batch_size,))
        nllh = self.crf._compute_numerator_log_likelihood(
            self.hidden, self.labels, self.mask
        )
        if torch.cuda.is_available():
            self.assertEqual(nllh.type(), "torch.cuda.FloatTensor")
        else:
            self.assertEqual(nllh.type(), "torch.FloatTensor")
        self.assertEqual(nllh.size(), (self.batch_size,))

    def test_decode(self):
        labels = self.crf.viterbi_decode(self.hidden, self.mask)
        self.assertEqual(type(labels), list)
        self.assertEqual(len(labels), self.batch_size)
        seq_lens = self.mask.sum(dim=1)
        self.assertEqual(len(labels[0]), seq_lens[0])
        self.assertEqual(len(labels[1]), seq_lens[1])
        label_types = list(set([int(label) for seq in self.labels for label in seq]))
        self.assertTrue([label in label_types for seq in labels for label in seq])

    def test_logsumexp(self):
        lse = self.crf.logsumexp(self.hidden, -1)
        self.assertEqual(lse.size(), (self.batch_size, self.sequence_size))


if __name__ == '__main__':
    unittest.main()