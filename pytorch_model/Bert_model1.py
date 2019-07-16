from __future__ import absolute_import, division, print_function, unicode_literals

"""
@autor:rentc

step1:bert-encoder --> sbj_vec
step2:sbj_vec-->s1,s2
step3:sbj_vec+ class_embedding -->obj_vec -->o1,o2

BERT + subj_Point +class+obj_point
"""

"""
spo_file:SPO数据
输入数据的格式：
[{"text": "123", "spo_list": [{"subject": "A", "relation_id": 5, "object": "B"}],{...}}]
text:文本描述
spo_list:关系描述 subject:主语 ，relation_id:关系ID ，object:宾语
"""

"""
relation_schemas_file:关系定义文件
输入数据的格式(每一行)(无表头)：relation_id\trelation
relation_id:关系ID
relation:关系
"""

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

    def __init__(self, config, num_class, max_len=128):
        super(MyBert, self).__init__(config)
        self.num_class = num_class
        self.max_len = max_len
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_sbj1 = nn.Linear(config.hidden_size, 2)
        self.classifier_sbj2 = nn.Linear(config.hidden_size, 2)
        self.classifier_obj1 = nn.Linear(config.hidden_size, 2)
        self.classifier_obj2 = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)
        self.class_emb = nn.Embedding(num_class, config.hidden_size)

        # CRF层
        # self.attcrf = AttnCRFDecoder.create(lable_size, self.emb_dim)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                s1=None, s2=None, k1=None, k2=None, relation_id=None, o1=None, o2=None):
        bert_last_layer, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        bert_last_layer = self.dropout(bert_last_layer)
        s1_vec = self.classifier_sbj1(bert_last_layer)
        s2_vec = self.classifier_sbj2(bert_last_layer)
        s1_vec = F.log_softmax(s1_vec.view(-1, self.max_len, 2), dim=-1)
        s2_vec = F.log_softmax(s2_vec.view(-1, self.max_len, 2), dim=-1)
        loss_s1 = (((-torch.sum(s1_vec * s1.view(-1, self.max_len, 2).float(), dim=-1))
                    * attention_mask.float()).sum(1)).mean()

        loss_s2 = (((-torch.sum(s2_vec * s2.view(-1, self.max_len, 2).float(), dim=-1))
                    * attention_mask.float()).sum(1)).mean()
        loss = loss_s1 + loss_s2
        if (k1 is None or k2 is None):
            return s1_vec, s2_vec, loss
        else:

            # k1,k2:sbj的开始位置 和结束位置
            c1 = []
            c2 = []
            k1 = k1.cpu().detach().numpy()
            for i, b in enumerate(bert_last_layer):
                c1.append(b[k1[i]])
                c2.append(b[k2[i]])
            t1 = torch.cat(c1).view(-1, 1, bert_last_layer.shape[-1])
            t2 = torch.cat(c2).view(-1, 1, bert_last_layer.shape[-1])
            rvec = self.class_emb(relation_id)
            t = (t1 + t2 + rvec.view(-1, 1, bert_last_layer.shape[-1])) / 3
            t = t * bert_last_layer

            pooled_output2 = self.dropout(t)
            o1_vec = self.classifier_obj1(pooled_output2)
            o2_vec = self.classifier_obj2(pooled_output2)

            o1_vec = F.log_softmax(o1_vec, dim=-1)
            o2_vec = F.log_softmax(o2_vec, dim=-1)
            loss_o1 = (((-torch.sum(o1_vec * o1.view(-1, self.max_len, 2).float(), dim=-1))
                        * attention_mask.float()).sum(1)).mean()

            loss_o2 = (((-torch.sum(o2_vec * o2.view(-1, self.max_len, 2).float(), dim=-1))
                        * attention_mask.float()).sum(1)).mean()

        return s1_vec, s2_vec, o1_vec, o2_vec, loss + loss_o1 + loss_o2


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
