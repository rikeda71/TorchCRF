import torch
import torch.nn as nn


# PAD = Padding
# BOS = Begin of State
# EOS = End of State
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
CUDA = torch.cuda.is_available()


class CRF(nn.Module):

    def __init__(self, num_labels: int) -> None:
        """

        :param num_labels: number of labels <int>
        """

        if num_labels < 1:
            raise ValueError('invalid number of labels: {0}'.format(num_labels))

        super().__init__()
        # PAD,BOS,EOSを追加
        self.num_labels = num_labels

        # 遷移行列の設定
        # 遷移行列のフォーマット [遷移先, 遷移元]
        # transition matrix settings
        # transition matrix format [destination, source]
        self.trans_matrix = nn.Parameter(self.myrandn(num_labels, num_labels))
        # BOSへは遷移しない
        # no transition to BOS
        self.trans_matrix.data[BOS_IDX, :] = -10000.
        # PADを除き，EOSとPADへは遷移しない
        # no transition from EOS and PAD except to PAD
        self.trans_matrix.data[:, EOS_IDX] = -10000.
        self.trans_matrix.data[:, PAD_IDX] = -10000.
        # EOSを除き，PADへは遷移しない
        # no transition to PAD except from EOS
        self.trans_matrix.data[PAD_IDX, :] = -10000.
        self.trans_matrix.data[PAD_IDX, EOS_IDX] = 0.
        self.trans_matrix.data[PAD_IDX, PAD_IDX] = 0.

    def forward(self, h, mask, batch_size: int):
        # ミニバッチ中のデータ全体の対数尤度を返す
        print(batch_size)
        score = self.myTensor(batch_size, self.num_labels).fill_(-10000.)
        score[:, BOS_IDX] = 0.
        # 計算できるよう，遷移行列のサイズを変更
        # [N, N] -> [1, batch_size, batch_size]
        trans = self.trans_matrix.unsqueeze(0)
        # ミニバッチ中の単語数だけ処理を行う
        # iterate through processing for the number of words in the mini batch
        for t in range(h.size(1)):
            # 各系列の系列のt番目のマスクを用意
            # prepare t-th mask of sequences in each sequence
            mask_t = mask[:, t].unsqueeze(1)
            # 各系列におけるt番目の系列ラベルの遷移確率
            # prepare the transition probability of the t-th sequence label in each sequence
            h_t = h[:, t].unsqueeze(-1)
            # 各系列でのt番目のスコアを導出
            # calculate t-th scores in each sequence
            print(mask_t.size())
            print(h_t.size())
            print(score.unsqueeze(1).size())
            score_t = self.logsumexp(score.unsqueeze(1) + h_t + trans)
            # スコアの更新
            # update scores
            score = score_t * mask_t + score * (1 - mask_t)
        # ミニバッチ中のデータ全体の対数尤度を返す
        # return the log likely food of all data in mini batch
        return self.logsumexp(score)

    def score(self, h, y, mask, batch_size: int):
        # Score(X,Y)の計算結果を返す
        score = self.myTensor(batch_size).fill_(0.)
        h = h.unsqueeze(-1)
        trans = self.trans_matrix.unsqueeze(-1)
        for t in range(h.size(1)):
            mask_t = mask[:, t]
            h_t = torch.cat([h[b, t, y[b, t + 1]] for b in range(batch_size)])
            trans_t = torch.cat(trans[s[t + 1], s[t]] for s in y)
            score += (h_t + trans_t) * mask_t
        return score

    def viterbi_decode(self, h, mask, batch_size: int):
        backpointer = self.myLongTensor()
        score = self.myTensor(batch_size, self.num_labels).fill_(-10000.)
        score[:, BOS_IDX] = 0.

        for t in range(h.size(1)):
            backpointer_t = self.myLongTensor()
            score_t = self.myTensor()
            for i in range(self.num_labels):  # for each next label
                m = [j.unsqueeze(1) for j in torch.max(score + self.trans_matrix[i], 1)]
                backpointer_t = torch.cat((backpointer_t, m[1]), 1)  # best previous labels
                score_t = torch.cat((score_t, m[0]), 1)  # best transition scores
            backpointer = torch.cat((backpointer, backpointer_t.unsqueeze(1)), 1)
            score = score_t + h[:, t]  # plus emission scores
        best_score, best_label = torch.max(score, 1)

        # back-tracking
        backpointer = backpointer.tolist()
        best_path = [[i] for i in best_label.tolist()]
        for b in range(batch_size):
            x = best_label[b]  # best label
            l = mask[b].sum().int().tolist()
            for b_t in reversed(backpointer[b][:l]):
                x = b_t[x]
                best_path[b].append(x)
            best_path[b].pop()
            best_path[b].reverse()

        return best_path


    @staticmethod
    def logsumexp(x):
        max = torch.max(x, -1)[0]
        return max + torch.log(torch.sum(torch.exp(x - max.unsqueeze(-1)), -1))

    @staticmethod
    def myTensor(*args):
        x = torch.Tensor(*args)
        return x.cuda() if CUDA else x

    @staticmethod
    def myLongTensor(*args):
        x = torch.LongTensor(*args)
        return x.cuda() if CUDA else x

    @staticmethod
    def myrandn(*args):
        x = torch.randn(*args)
        return x.cuda() if CUDA else x
