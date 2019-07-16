"""
@autor:rentc

step1:bert-encoder --> sbj_vec
step2:sbj_vec-->s1,s2
step3:sbj_vec+ class_embedding -->obj_vec -->o1,o2

BERT + subj_Point +class+obj_point
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from keras_bert import Tokenizer
import os
import codecs
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel, BertForSequenceClassification, BertConfig, \
    WEIGHTS_NAME, CONFIG_NAME
import copy
import json
import logging
import math
import os
import shutil
import tarfile
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class MyBert(BertPreTrainedModel):

    def __init__(self, config, num_class, max_len=128, use_fp16=True):
        super(MyBert, self).__init__(config)
        self.num_class = num_class
        self.max_len = max_len
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_sbj1 = nn.Linear(config.hidden_size, 1)
        self.classifier_sbj2 = nn.Linear(config.hidden_size, 1)
        self.classifier_obj1 = nn.Linear(config.hidden_size, num_class)
        self.classifier_obj2 = nn.Linear(config.hidden_size, num_class)
        self.apply(self.init_bert_weights)
        self.class_emb = nn.Embedding(num_class, config.hidden_size)
        self.lossfun = nn.BCELoss()
        self.lossfun1 = nn.BCEWithLogitsLoss(reduction="none")
        self.use_fp16 = use_fp16
        # CRF层
        # self.attcrf = AttnCRFDecoder.create(lable_size, self.emb_dim)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                s1=None, s2=None, k1=None, k2=None, o1=None, o2=None):
        bert_last_layer, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        bert_last_layer = self.dropout(bert_last_layer)
        s1_vec = self.classifier_sbj1(bert_last_layer)
        s2_vec = self.classifier_sbj2(bert_last_layer)
        s1_p = torch.sigmoid(s1_vec.view(-1, self.max_len))
        s2_p = torch.sigmoid(s2_vec.view(-1, self.max_len))
        if ((s1 is None or s2 is None) and (k1 is None or k2 is None)):
            return s1_p, s2_p

        if (self.use_fp16):
            mask = attention_mask.float().half()
        else:
            mask = attention_mask.float()
        if (s1 is not None or s2 is not None):
            if (self.use_fp16):
                s1 = s1.float().half()
                s2 = s2.float().half()
            else:
                s1 = s1.float()
                s2 = s2.float()

            if (self.use_fp16):
                r = self.lossfun1(s1_vec.view(-1, self.max_len), s1) * mask
                loss_s1 = torch.mean(r.sum(-1))
                r = self.lossfun1(s2_vec.view(-1, self.max_len), s2) * mask
                loss_s2 = torch.mean(r.sum(-1))

            else:
                loss_s1 = torch.sum(
                    -(torch.log(s1_p) * s1 + torch.log(1.000001 - s1_p) * (1 - s1)) * mask,
                    dim=-1).mean()
                loss_s2 = torch.sum(
                    -(torch.log(s2_p) * s2 + torch.log(1.000001 - s2_p) * (1 - s2)) * mask,
                    dim=-1).mean()

            loss = loss_s1 + loss_s2
        if (o1 is not None or o2 is not None):
            o1 = o1.float()
            o2 = o2.float()
        # k1,k2:sbj的开始位置 和结束位置
        c1 = []
        c2 = []
        k1 = k1.cpu().detach().numpy()
        for i, b in enumerate(bert_last_layer):
            c1.append(b[k1[i]])
            c2.append(b[k2[i]])
        t1 = torch.cat(c1).view(-1, 1, bert_last_layer.shape[-1])
        t2 = torch.cat(c2).view(-1, 1, bert_last_layer.shape[-1])
        t = (t1 + t2) / 2
        t = t * bert_last_layer

        pooled_output2 = self.dropout(t)
        o1_vec = self.classifier_obj1(pooled_output2)
        o2_vec = self.classifier_obj2(pooled_output2)
        o1_p = torch.sigmoid(o1_vec).transpose(1, 2)
        o2_p = torch.sigmoid(o2_vec).transpose(1, 2)
        if (o1 is None or o2 is None):
            return o1_p, o2_p

        if (self.use_fp16):
            o1 = o1.float().half()
            o2 = o2.float().half()

        if (self.use_fp16):
            r = self.lossfun1(o1_vec.transpose(1, 2), o1)
            loss_o1 = torch.mean((mask.view(-1, 1, self.max_len) * r).sum(-1).sum(-1))

            r = self.lossfun1(o2_vec.transpose(1, 2), o2)
            loss_o2 = torch.mean((mask.view(-1, 1, self.max_len) * r).sum(-1).sum(-1))

        else:
            loss_o1 = (
                torch.sum(-mask.view(-1, 1, self.max_len) * (
                        torch.log(o1_p) * o1 + torch.log(1.000001 - o1_p) * (1 - o1)),
                          dim=-1)
                    .sum(dim=-1)  # mean
            ).mean()

            loss_o2 = (
                torch.sum(-mask.view(-1, 1, self.max_len) * (
                        torch.log(o2_p) * o2 + torch.log(1.000001 - o2_p) * (1 - o2)),
                          dim=-1).sum(dim=-1)).mean()

        return s1_p, s2_p, o1_p, o2_p, loss + loss_o1 + loss_o2


class MyTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


def init_token_dict(dict_path):
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


def init_relation(Relation_ID_PATH):
    id2relation = {}
    relation2id = {}
    with open(Relation_ID_PATH) as f:
        for l in f:
            s = l.strip().split("\t")
            id = (int)(s[0])
            relation = s[1]
            id2relation[id] = relation
            relation2id[relation] = id
    return id2relation, relation2id

# grads_groups_flat.append(
#                 _flatten_dense_tensors([p.grad if p.grad is not None else p.new_zeros(p.size()) for p in group]))
