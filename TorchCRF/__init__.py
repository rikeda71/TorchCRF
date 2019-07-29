from typing import List
import torch
import torch.nn as nn


class CRF(nn.Module):
    CUDA = torch.cuda.is_available()

    def __init__(self, num_labels: int, pad_idx: int = None) -> None:
        """

        :param num_labels: number of labels
        :param pad_idxL padding index. default None
        :return None
        """

        if num_labels < 1:
            raise ValueError(
                'invalid number of labels: {0}'.format(num_labels))

        super().__init__()
        self.num_labels = num_labels

        # 遷移行列の設定
        # 遷移行列のフォーマット (遷移元, 遷移先)
        # transition matrix setting
        # transition matrix format (source, destination)
        self.trans_matrix = nn.Parameter(self.myTensor(num_labels, num_labels))
        # 先頭と末尾への遷移行列の設定
        # transition matrix of start and end settings
        self.start_trans = nn.Parameter(self.myTensor(num_labels))
        self.end_trans = nn.Parameter(self.myTensor(num_labels))

        self._initialize_parameters()

        if pad_idx is not None:
            self.start_trans[pad_idx] = -10000.
            self.trans_matrix[pad_idx, :] = -10000.
            self.trans_matrix[:, pad_idx] = -10000.
            self.trans_matrix[pad_idx, pad_idx] = 0.

    def forward(self, h: torch.FloatTensor,
                labels: torch.LongTensor,
                mask: torch.FloatTensor) -> torch.FloatTensor:
        """

        :param h: hidden matrix (seq_len, batch_size, num_labels)
        :param labels: answer labels of each sequence
                       in mini batch (seq_len, batch_size)
        :param mask: mask tensor of each sequence
                     in mini batch (seq_len, batch_size)
        :return: The log-likelihood (batch_size)
        """

        log_numerator = self._compute_numerator_log_likelihood(h, labels, mask)
        log_denominator = self._compute_denominator_log_likelihood(h, mask)

        return log_numerator - log_denominator

    def viterbi_decode(self, h: torch.FloatTensor,
                       mask: torch.FloatTensor) -> List[List[int]]:
        """
        decode labels using viterbi algorithm
        :param h: hidden matrix (batch_size, seq_len, num_labels)
        :param mask: mask tensor of each sequence
                     in mini batch (seq_len, batch_size)
        :return: labels of each sequence in mini batch
        """

        batch_size, seq_len, _ = h.size()
        # 各系列の系列長を用意
        # prepare the sequence lengths in each sequence
        seq_lens = mask.long().sum(dim=1)
        # バッチ内において，スタート地点から先頭のラベルに対してのスコアを用意
        # In mini batch, prepare the score
        # from the start sequence to the first label
        score = [self.start_trans.data + h[:, 0]]
        path = []

        for t in range(1, seq_len):
            # 1つ前の系列のスコアを抽出
            # extract the score of previous sequence
            # (batch_size, num_labels, 1)
            previous_score = score[t - 1].view(batch_size, -1, 1)

            # 系列の隠れ層のスコアを抽出
            # extract the score of hidden matrix of sequence
            # (batch_size, 1, num_labels)
            h_t = h[:, t].view(batch_size, 1, -1)

            # t-1の系列のラベルからtの系列のラベルまでの遷移におけるスコアを抽出
            # self.trans_matrixは系列Aから系列Bまでの遷移のスコアを持っている
            # extract the score in transition
            # from label of t-1 sequence to label of sequence of t
            # self.trans_matrix has the score of the transition
            # from sequence A to sequence B
            # (batch_size, num_labels, num_labels)
            score_t = previous_score + self.trans_matrix + h_t

            # 導出したスコアのうち，各系列の最大値と最大値をとり得る位置を保持
            # keep the maximum value
            # and point where maximum value of each sequence
            # (batch_size, num_labels)
            best_score, best_path = score_t.max(1)
            score.append(best_score)
            path.append(best_path)

        # バッチ内のラベルを推定
        # predict labels of mini batch
        best_paths = [self._viterbi_compute_best_path(i, seq_lens, score, path)
                      for i in range(batch_size)]

        return best_paths

    def _viterbi_compute_best_path(self, batch_idx: int,
                                   seq_lens: torch.LongTensor,
                                   score: List[torch.FloatTensor],
                                   path: List[torch.LongTensor]) -> List[int]:
        """
        return labels using viterbi algorithm
        :param batch_idx: index of batch
        :param seq_lens: sequence lengths in mini batch (batch_size)
        :param score: transition scores of length max sequence size
                      in mini batch [(batch_size, num_labels)]
        :param path: transition paths of length max sequence size
                     in mini batch [(batch_size, num_labels)]
        :return: labels of batch_idx-th sequence
        """

        seq_end_idx = seq_lens[batch_idx] - 1
        # 系列の一番後ろのラベルを抽出
        # extract label of end sequence
        _, best_last_label = \
            (score[seq_end_idx][batch_idx] + self.end_trans).max(0)
        best_labels = [int(best_last_label)]

        # viterbiアルゴリズムにより，ラベルを後ろから推定
        # predict labels from back using viterbi algorithm
        for p in reversed(path[: seq_end_idx]):
            best_last_label = p[batch_idx][best_labels[0]]
            best_labels.insert(0, int(best_last_label))

        return best_labels

    def _compute_denominator_log_likelihood(self, h: torch.FloatTensor,
                                            mask: torch.FloatTensor):
        """

        compute the denominator term for the log-likelihood
        :param h: hidden matrix (batch_size, seq_len, num_labels)
        :param mask: mask tensor of each sequence
                     in mini batch (batch_size, seq_len)
        :return: The score of denominator term for the log-likelihood
        """

        batch_size, seq_len, _ = h.size()
        # 計算できるよう，遷移行列のサイズを変更
        # (num_labels, num_labels) -> (1, num_labels, num_labels)
        trans = self.trans_matrix.view(1, self.num_labels, self.num_labels)
        # 先頭から各ラベルへのスコアと各ラベルの1番目のスコアを足し合わせる
        # add the score from beginning to each label
        # and the first score of each label
        score = self.start_trans.view(1, -1) + h[:, 0]
        # ミニバッチ中の単語数だけ処理を行う
        # iterate through processing for the number of words in the mini batch
        for t in range(1, seq_len):
            # (batch_size, self.num_labels, 1)
            before_score = score.view(batch_size, self.num_labels, 1)
            # 各系列の系列のt番目のマスクを用意
            # prepare t-th mask of sequences in each sequence
            # (batch_size, 1)
            mask_t = mask[:, t].view(batch_size, 1)
            # 各系列におけるt番目の系列ラベルの遷移確率
            # prepare the transition probability of the t-th sequence label
            # in each sequence
            # (batch_size, 1, num_labels)
            h_t = h[:, t].view(batch_size, 1, self.num_labels)
            # 各系列でのt番目のスコアを導出
            # calculate t-th scores in each sequence
            # (batch_size, num_labels)
            score_t = self.logsumexp(before_score + h_t + trans, 1)
            # スコアの更新
            # update scores
            # (batch_size, num_labels)
            score = score_t * mask_t + score * (1 - mask_t)

        # 末尾のスコアを足し合わせる
        # add the end score of each label
        score += self.end_trans.view(1, -1)
        # ミニバッチ中のデータ全体の対数尤度を返す
        # return the log likely food of all data in mini batch
        return self.logsumexp(score, 1)

    def _compute_numerator_log_likelihood(
            self, h: torch.FloatTensor,
            y: torch.LongTensor,
            mask: torch.FloatTensor) -> torch.FloatTensor:
        """
        compute the numerator term for the log-likelihood
        :param h: hidden matrix (batch_size, seq_len, num_labels)
        :param y: answer labels of each sequence
                  in mini batch (batch_size, seq_len)
        :param mask: mask tensor of each sequence
                     in mini batch (batch_size, seq_len)
        :return: The score of numerator term for the log-likelihood
        """

        batch_size, seq_len, _ = h.size()
        # 系列のスタート位置のベクトルを抽出
        # extract first vector of sequences in mini batch
        score = self.start_trans[y[:, 0]]

        h = h.unsqueeze(-1)
        trans = self.trans_matrix.unsqueeze(-1)

        for t in range(seq_len - 1):
            mask_t = mask[:, t]
            mask_t1 = mask[:, t + 1]
            # t+1番目のラベルのスコアを抽出
            # extract the score of t+1 label
            # (batch_size)
            h_t = torch.cat([h[b, t, y[b, t]] for b in range(batch_size)])
            # t番目のラベルからt+1番目のラベルへの遷移スコアを抽出
            # extract the transition score from t-th label to t+1 label
            # (batch_size)
            trans_t = torch.cat([trans[s[t], s[t + 1]] for s in y])
            # 足し合わせる
            # add the score of t+1 and the transition score
            # (batch_size)
            score += h_t * mask_t + trans_t * mask_t1

        # バッチ内の各系列の最後尾のラベル番号を抽出する
        # extract end label number of each sequence in mini batch
        # (batch_size)
        last_mask_index = mask.long().sum(1) - 1
        last_labels = y.gather(1, last_mask_index.unsqueeze(-1))
        # hの形を元に戻す
        # restore the shape of h
        h = h.unsqueeze(-1).view(batch_size, seq_len, self.num_labels)

        # バッチ内の最大長の系列のスコアを足し合わせる
        # Add the score of the sequences of the maximum length in mini batch
        score += h[:, -1].gather(1, last_labels).squeeze(1) * mask[:, -1]
        # 各系列の最後尾のタグからEOSまでのスコアを足し合わせる
        # Add the scores from the last tag of each sequence to EOS
        score += self.end_trans[last_labels].view(batch_size)

        return score

    def _initialize_parameters(self) -> None:
        """
        initialize transition parameters
        :return: None
        """

        nn.init.uniform_(self.trans_matrix, -0.1, 0.1)
        nn.init.uniform_(self.start_trans, -0.1, 0.1)
        nn.init.uniform_(self.end_trans, -0.1, 0.1)

    @staticmethod
    def logsumexp(x: torch.FloatTensor, dim: int) -> torch.FloatTensor:
        """
        return log(sum(exp(x))) while minimizing
                                the possibility of overflow/underflow.
        :param x: the matrix format torch.FloatTensor
        :param dim: dimensiton
        :return: log(sum(exp(x)))
        """

        vmax, _ = x.max(dim)
        return vmax + \
            torch.log(torch.sum(torch.exp(x - vmax.unsqueeze(dim)), dim))

    @staticmethod
    def myTensor(*args) -> torch.Tensor:
        x = torch.Tensor(*args)
        return x.cuda() if CRF.CUDA else x

    @staticmethod
    def myLongTensor(*args) -> torch.LongTensor:
        x = torch.LongTensor(*args)
        return x.cuda() if CRF.CUDA else x

    @staticmethod
    def myrandn(*args) -> torch.Tensor:
        x = torch.randn(*args)
        return x.cuda() if CRF.CUDA else x
