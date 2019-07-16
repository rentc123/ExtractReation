import torch
from torch import nn
import numpy as np
from torch.nn import init

MaxLen = 256


# TODO: move to utils
def log_sum_exp(tensor, dim=0):
    """LogSumExp operation."""
    m, _ = torch.max(tensor, dim)
    m_exp = m.unsqueeze(-1).expand_as(tensor)
    return m + torch.log(torch.sum(torch.exp(tensor - m_exp), dim))


def sequence_mask(lens, max_len=None):
    batch_size = lens.size(0)

    if max_len is None:
        max_len = lens.max().item()

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, (int)(max_len))

    if lens.data.is_cuda:
        ranges = ranges.cuda()

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp.long()

    return mask


class CRF(nn.Module):
    def __init__(self, label_size):
        super(CRF, self).__init__()

        self.label_size = label_size
        self.start = self.label_size - 2
        self.end = self.label_size - 1
        transition = torch.randn(self.label_size, self.label_size)
        self.transition = nn.Parameter(transition)
        self.initialize()

    def initialize(self):
        self.transition.data[:, self.end] = -100.0
        self.transition.data[self.start, :] = -100.0

    def pad_logits(self, logits):
        # lens = lens.data
        batch_size, seq_len, label_num = logits.size()
        # pads = Variable(logits.data.new(batch_size, seq_len, 2).fill_(-1000.0),
        #                 requires_grad=False)
        pads = logits.new_full((batch_size, seq_len, 2), -1000.0,
                               requires_grad=False)
        logits = torch.cat([logits, pads], dim=2)
        return logits

    def calc_binary_score(self, labels, lens):
        batch_size, seq_len = labels.size()

        # labels_ext = Variable(labels.data.new(batch_size, seq_len + 2))
        labels_ext = labels.new_empty((batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, max_len=(seq_len + 2)).long()
        # pad_stop = Variable(labels.data.new(1).fill_(self.end))
        pad_stop = labels.new_full((1,), self.end, requires_grad=False)
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        trn = self.transition
        trn_exp = trn.unsqueeze(0).expand(batch_size, *trn.size())
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), trn.size(0))
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(lens + 1, max_len=MaxLen + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr

        return score

    def calc_unary_score(self, logits, labels, lens):
        labels_exp = labels.unsqueeze(-1)
        scores = torch.gather(logits, 2, labels_exp).squeeze(-1)
        mask = sequence_mask(lens, max_len=MaxLen).float()
        scores = scores * mask
        return scores

    def calc_gold_score(self, logits, labels, lens):
        unary_score = self.calc_unary_score(logits, labels, lens).sum(
            1).squeeze(-1)
        binary_score = self.calc_binary_score(labels, lens).sum(1).squeeze(-1)
        return unary_score + binary_score

    def calc_norm_score(self, logits, lens):
        batch_size, seq_len, feat_dim = logits.size()
        # alpha = logits.data.new(batch_size, self.label_size).fill_(-10000.0)
        alpha = logits.new_full((batch_size, self.label_size), -100.0)
        alpha[:, self.start] = 0
        # alpha = Variable(alpha)
        lens_ = lens.clone()

        logits_t = logits.transpose(1, 0)
        for logit in logits_t:
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   *self.transition.size())
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  *self.transition.size())
            trans_exp = self.transition.unsqueeze(0).expand_as(alpha_exp)
            mat = logit_exp + alpha_exp + trans_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)

            mask = (lens_ > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            lens_ = lens_ - 1

        alpha = alpha + self.transition[self.end].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def viterbi_decode(self, logits, lens):
        """Borrowed from pytorch tutorial
        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, seq_len, n_labels = logits.size()
        # vit = logits.data.new(batch_size, self.label_size).fill_(-10000)
        vit = logits.new_full((batch_size, self.label_size), -100.0)
        vit[:, self.start] = 0
        # vit = Variable(vit)
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        pointers = []
        for logit in logits_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transition.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transition[self.end].unsqueeze(
                0).expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        # idx = idx.squeeze(-1)
        paths = [idx.unsqueeze(1)]
        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths


class AttnCRFDecoder(nn.Module):
    def __init__(self,
                 crf, label_size, input_dim, input_dropout=0.5,
                 key_dim=64, val_dim=64, num_heads=3):
        super(AttnCRFDecoder, self).__init__()
        self.input_dim = input_dim
        self.attn = MultiHeadAttention(key_dim, val_dim, input_dim, num_heads, input_dropout)
        self.linear = Linears(in_features=input_dim,
                              out_features=label_size,
                              hiddens=[input_dim // 2])
        self.crf = crf
        self.label_size = label_size

    def forward_model(self, inputs, labels_mask=None):
        batch_size, seq_len, input_dim = inputs.size()
        inputs, _ = self.attn(inputs, inputs, inputs, labels_mask)

        output = inputs.contiguous().view(-1, self.input_dim)
        # Fully-connected layer
        output = self.linear.forward(output)
        output = output.view(batch_size, seq_len, self.label_size)
        return output

    def forward(self, inputs, labels_mask):
        self.eval()
        lens = labels_mask.sum(-1)
        logits = self.forward_model(inputs)
        logits = self.crf.pad_logits(logits)
        scores, preds = self.crf.viterbi_decode(logits, lens)
        self.train()
        return preds

    def score(self, inputs, labels_mask, labels):
        lens = labels_mask.sum(-1)
        # ————————源码中这一步没有传mask参数，Attention是全部计算————————
        logits = self.forward_model(inputs)
        logits = self.crf.pad_logits(logits)
        norm_score = self.crf.calc_norm_score(logits, lens)
        gold_score = self.crf.calc_gold_score(logits, labels, lens)
        loglik = gold_score - norm_score
        return -loglik.mean()

    @classmethod
    def create(cls, label_size, input_dim, input_dropout=0.5, key_dim=64, val_dim=64, num_heads=3):
        return cls(CRF(label_size + 2), label_size, input_dim, input_dropout,
                   key_dim, val_dim, num_heads)


from torch.nn import functional
import numpy as np
import torch
from torch import nn
from torch.nn import init
import math


class Linear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True):
        super(Linear, self).__init__(in_features, out_features, bias=bias)
        init.orthogonal_(self.weight)


class Linears(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hiddens,
                 bias=True,
                 activation='tanh'):
        super(Linears, self).__init__()
        assert len(hiddens) > 0

        self.in_features = in_features
        self.out_features = self.output_size = out_features

        in_dims = [in_features] + hiddens[:-1]
        self.linears = nn.ModuleList([Linear(in_dim, out_dim, bias=bias)
                                      for in_dim, out_dim
                                      in zip(in_dims, hiddens)])
        self.output_linear = Linear(hiddens[-1], out_features, bias=bias)
        self.activation = getattr(torch, activation)

    def forward(self, inputs):
        linear_outputs = inputs
        for linear in self.linears:
            linear_outputs = linear.forward(linear_outputs)
            linear_outputs = self.activation(linear_outputs)
        return self.output_linear.forward(linear_outputs)


# Reused from https://github.com/JayParks/transformer/
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x len_q x d_k]
        # k: [b_size x len_k x d_k]
        # v: [b_size x len_v x d_v] note: (len_k == len_v)
        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale_factor  # attn: [b_size x len_q x len_k]
        if attn_mask is not None:
            assert attn_mask.size() == attn.size()
            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        outputs = torch.bmm(attn, v)  # outputs: [b_size x len_q x d_v]

        return outputs, attn


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(d_hid), requires_grad=True)
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True, )
        std = z.std(dim=-1, keepdim=True, )
        ln_out = (z - mean.expand_as(z)) / (std.expand_as(z) + self.eps)
        ln_out = self.gamma.expand_as(ln_out) * ln_out + self.beta.expand_as(ln_out)

        return ln_out


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        self.w_q = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_k))
        self.w_k = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_k))
        self.w_v = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_k, dropout)

        init.xavier_normal_(self.w_q)
        init.xavier_normal_(self.w_k)
        init.xavier_normal_(self.w_v)

    def forward(self, q, k, v, attn_mask=None):
        (d_k, d_v, d_model, n_heads) = (self.d_k, self.d_v, self.d_model, self.n_heads)
        b_size = k.size(0)

        q_s = q.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_q x d_model]
        k_s = k.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_k x d_model]
        v_s = v.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_v x d_model]

        q_s = torch.bmm(q_s, self.w_q).view(b_size * n_heads, -1, d_k)  # [b_size * n_heads x len_q x d_k]
        k_s = torch.bmm(k_s, self.w_k).view(b_size * n_heads, -1, d_k)  # [b_size * n_heads x len_k x d_k]
        v_s = torch.bmm(v_s, self.w_v).view(b_size * n_heads, -1, d_v)  # [b_size * n_heads x len_v x d_v]

        # perform attention, result_size = [b_size * n_heads x len_q x d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_heads, 1, 1)
        outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)

        # return a list of tensors of shape [b_size x len_q x d_v] (length: n_heads)
        return torch.split(outputs, b_size, dim=0), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.attention = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.proj = Linear(n_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        residual = q
        # outputs: a list of tensors of shape [b_size x len_q x d_v] (length: n_heads)
        outputs, attn = self.attention(q, k, v, attn_mask=attn_mask)
        # concatenate 'n_heads' multi-head attentions
        outputs = torch.cat(outputs, dim=-1)
        # project back to residual size, result_size = [b_size x len_q x d_model]
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        return self.layer_norm(residual + outputs), attn





        #
        # batch_size=2
        # len_seq=3
        # input_dim=10
        # lable_size=5
        #
        # attcrf=AttnCRFDecoder.create(lable_size,input_dim)
        # inputs=torch.rand(batch_size,len_seq,input_dim)
        # masks=torch.ones(batch_size,len_seq)
        # labels=torch.ones(batch_size,len_seq)
        # labels=labels.long()
        # attcrf(inputs,masks)
        # attcrf.score(inputs,masks,labels)
