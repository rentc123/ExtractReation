from __future__ import absolute_import, division, print_function
from keras_bert import Tokenizer
import os
import numpy as np
import pickle
import codecs
import json
from apex.optimizers import FP16_Optimizer
from apex.optimizers import FusedAdam
from pytorch_pretrained_bert.optimization import WarmupLinearSchedule, BertAdam
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, RandomSampler, DistributedSampler, DataLoader
from tqdm import trange, tqdm

from pytorch_model.Bert_model1 import MyBert
import torch
import csv
import logging
import os
import sys
from pytorch_model.Bert_model1 import init_token_dict, MyTokenizer, init_relation

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MAX_LEN = 180
do_train = True
do_eval = False
do_inference = True
shuffle = True
lr = 5e-5
batch_size = 32
num_train_epochs = 3
fp16 = False
loss_scale = 0
warmup_proportion = 0.1

VACB_PATH = "/home/rentc/project/bert_chinese_model/bert-base-chinese-vocab.txt"
BERT_PRETRAINED_PATH = "/home/rentc/project/bert_chinese_model/download_bert-base-chinese.tar.gz"

data_dir = "/home/rentc/project/比赛/2019CCF/信息抽取/data/after_process"
Relation_ID_PATH = "/home/rentc/project/比赛/2019CCF/信息抽取/data/after_process/relation2id.csv"

id2relation, relation2id = {}, {}
id2relation, relation2id = init_relation(Relation_ID_PATH)
num_class = len(id2relation)
token_dict = init_token_dict(VACB_PATH)
tokenizer = MyTokenizer(token_dict)
no_cuda = False
model = MyBert.from_pretrained(BERT_PRETRAINED_PATH, num_class=num_class, max_len=MAX_LEN)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, sbj=None, obj=None, relation_id=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.sbj = sbj
        self.obj = obj
        self.relation_id = relation_id


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, s1, s2, k1, k2, o1, o2, relation_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.s1 = s1
        self.s2 = s2
        self.k1 = k1
        self.k2 = k2
        self.o1 = o1
        self.o2 = o2
        self.relation_id = relation_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MyProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            json.load(open(os.path.join(data_dir, "train.json"))), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            json.load(open(os.path.join(data_dir, "dev.json"))), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, str(i))
            text = line["text"]
            spo_list = line["spo_list"]
            for spo in spo_list:
                sbj = spo["subject"]
                obj = spo["object"]
                relation_id = spo["relation_id"]
                examples.append(
                    InputExample(guid=guid, text=text, sbj=sbj, obj=obj, relation_id=relation_id))
        return examples


processor = MyProcessor()
train_examples = processor.get_train_examples(data_dir)
dev_examples = processor.get_dev_examples(data_dir)
device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")


def process_examples_to_features(examples, MAX_LEN, tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        text = example.text[:MAX_LEN - 2]
        sbj = example.sbj
        obj = example.obj
        relation_id = example.relation_id
        input_id, token_type_id = tokenizer.encode(first=text, max_len=MAX_LEN)

        # input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
        # input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
        # token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
        attention_mask = [1] * (len(text) + 2)
        attention_mask += [0] * (MAX_LEN - len(attention_mask))
        # attention_mask = torch.LongTensor([attention_mask])

        # relation_id = 2
        # relation_id = torch.LongTensor([relation_id])

        # text = "阿斯蒂芬岁离开了"
        # input_id, token_type_id = tokenizer.encode(first=text, max_len=MAX_LEN)

        # input_ids = torch.LongTensor([input_id])
        # token_type_ids = torch.LongTensor([token_type_id])
        s1 = text.find(sbj)
        s2 = -1
        if (s1 != -1):
            s1 += 1  # 起始位置多一个CLS
            s2 = s1 + len(sbj) - 1

        o1 = text.find(obj)
        o2 = -1
        if (o1 != -1):
            o1 += 1  # 起始位置多一个CLS
            o2 = o1 + len(obj) - 1

        if (s1 != -1 and o1 != -1 and s2 < MAX_LEN and o2 < MAX_LEN):
            ts1 = [1, 0] * MAX_LEN
            ts1[2 * s1] = 0
            ts1[2 * s1 + 1] = 1
            s1_ = ts1
            ts2 = [1, 0] * MAX_LEN
            ts2[2 * s2] = 0
            ts2[2 * s2 + 1] = 1
            s2_ = ts2

            # o1_, o2_ = np.zeros((num_class, MAX_LEN * 2)), np.zeros((num_class, MAX_LEN * 2))
            # o1_[relation_id][2 * o1] = 1
            # o2_[relation_id][2 * o2 + 1] = 1

            to1 = [1, 0] * MAX_LEN
            to1[2 * o1] = 0
            to1[2 * o1 + 1] = 1
            o1_ = to1
            to2 = [1, 0] * MAX_LEN
            to2[2 * o2] = 0
            to2[2 * o2 + 1] = 1
            o2_ = to2

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokenizer.tokenize(text)]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_id]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in token_type_id]))
            logger.info("relation_id: %s " % (example.relation_id))
            logger.info("subject: %s " % (sbj))
            logger.info("s1: %s " % (s1))
            logger.info("s2: %s " % (s2))
            logger.info("object: %s " % (obj))
            logger.info("o1: %s " % (o1))
            logger.info("o2: %s " % (o2))

        # input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, s1 = s1, s2 = s2,
        # o1 = o1, o2 = o2, relation_id = relation_id

        assert len(attention_mask) == MAX_LEN

        features.append(
            InputFeatures(input_ids=input_id,
                          input_mask=attention_mask,
                          segment_ids=token_type_id,
                          s1=s1_, s2=s2_, k1=s1, k2=s2, o1=o1_, o2=o2_,
                          relation_id=relation_id))

    return features


def create_dataloder(train_examples):
    # Prepare data loader
    train_features = process_examples_to_features(train_examples, MAX_LEN, tokenizer, )

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_s1s = torch.tensor([f.s1 for f in train_features], dtype=torch.long)
    all_s2s = torch.tensor([f.s2 for f in train_features], dtype=torch.long)
    all_k1s = torch.tensor([f.k1 for f in train_features], dtype=torch.long)
    all_k2s = torch.tensor([f.k2 for f in train_features], dtype=torch.long)
    all_o1s = torch.tensor([f.o1 for f in train_features], dtype=torch.long)
    all_o2s = torch.tensor([f.o2 for f in train_features], dtype=torch.long)
    all_relation_ids = torch.tensor([f.relation_id for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                               all_s1s, all_s2s, all_k1s, all_k2s, all_o1s, all_o2s, all_relation_ids)
    if shuffle == True:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return dataloader


n_gpu = torch.cuda.device_count()

model.to(device)


def print_train_metrics(s1_vec, s2_vec, s1, s2, input_mask,
                        o1_vec=None, o2_vec=None, o1=None, o2=None):
    from sklearn.metrics import f1_score
    s1_p = (torch.exp(s1_vec)[:, :, 1] * input_mask.float()).cpu().detach().numpy()
    s1 = (s1.view(-1, MAX_LEN, 2)[:, :, 1]).cpu().detach().numpy()

    s2_p = (torch.exp(s2_vec)[:, :, 1] * input_mask.float()).cpu().detach().numpy()
    s2 = (s2.view(-1, MAX_LEN, 2)[:, :, 1]).cpu().detach().numpy()

    o1_p = (torch.exp(o1_vec)[:, :, 1] * input_mask.float()).cpu().detach().numpy()
    o1 = (o1.view(-1, MAX_LEN, 2)[:, :, 1]).cpu().detach().numpy()

    o2_p = (torch.exp(o2_vec)[:, :, 1] * input_mask.float()).cpu().detach().numpy()
    o2 = (o2.view(-1, MAX_LEN, 2)[:, :, 1]).cpu().detach().numpy()
    f1score_sbj = 0
    f1score_obj = 0
    f1 = 0
    correct = 0
    for i in range(s1_p.shape[0]):

        s1_pre = np.where(s1_p[i] > 0.5)[0]
        s1_y = np.where(s1[i] == 1)[0]
        s2_pre = np.where(s2_p[i] > 0.5)[0]
        s2_y = np.where(s2[i] == 1)[0]
        sbj_list = []
        for k1 in s1_pre:
            k2 = s2_pre[s2_pre > k1]
            if (len(k2) > 0):
                sbj_list.append((k1, k2[0]))

        o1_pre = np.where(o1_p[i] > 0.5)[0]
        o1_y = np.where(o1[i] == 1)[0]
        o2_pre = np.where(o2_p[i] > 0.5)[0]
        o2_y = np.where(o2[i] == 1)[0]
        obj_list = []
        for k1 in o1_pre:
            k2 = o2_pre[o2_pre > k1]
            if (len(k2) > 0):
                obj_list.append((k1, k2[0]))

        so_y = (s1_y[0], s2_y[0], o1_y[0], o2_y[0])

        for sbj in sbj_list:
            for obj in obj_list:
                if ((sbj[0], sbj[1], obj[0], obj[1]) == so_y):
                    correct += 1
        precise = 0
        if ((len(sbj_list) * len(obj_list)) != 0):
            precise = correct / (len(sbj_list) * len(obj_list))
        f1 += 2 * 1 * precise / (precise + 1)

        pre1 = np.zeros_like(s1[i])
        pre1[s1_p[i] > 0.5] = 1
        y1 = s1[i]

        pre2 = np.zeros_like(s2[i])
        pre2[s2_p[i] > 0.5] = 1
        y2 = s2[i]

        f1_s1 = f1_score(pre1, y1)
        f1_s2 = f1_score(pre2, y2)

        print(f1_s1, f1_s2)
        f1score_sbj += (f1_s1 + f1_s2) / 2

        pre1 = np.zeros_like(o1[i])
        pre1[o1_p[i] > 0.5] = 1
        y1 = o1[i]
        pre2 = np.zeros_like(o2[i])
        pre2[o2_p[i] > 0.5] = 1
        y2 = o2[i]
        f1_s1 = f1_score(pre1, y1)
        f1_s2 = f1_score(pre2, y2)

        f1score_obj += (f1_s1 + f1_s2) / 2

    logger.info("sbj f1_score:{}".format(f1score_sbj / s1_p.shape[0]))
    logger.info("obj f1_score:{}".format(f1score_obj / s1_p.shape[0]))
    logger.info("f1_score:{},correct_percent:{}".format(f1 / s1_p.shape[0], correct / s1_p.shape[0]))


if do_train:
    tb_writer = SummaryWriter()
    train_data_loader_path = "dataloader/train_data_loader.pkl"
    if (os.path.exists(train_data_loader_path)):
        train_dataloader = pickle.load(open("dataloader/train_data_loader.pkl", 'rb'))
    else:
        train_dataloader = create_dataloder(train_examples)
        pickle.dump(train_dataloader, open(train_data_loader_path, 'wb'))

    model.train()

    num_train_optimization_steps = len(train_dataloader) // num_train_epochs

    # Prepare optimizer

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if fp16:
        optimizer = FusedAdam(optimizer_grouped_parameters, lr=lr, bias_correction=False, max_grad_norm=1.0)
        if loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=warmup_proportion, t_total=num_train_optimization_steps)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=lr,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    model.train()
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    for _ in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", )):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, s1, s2, k1, k2, o1, o2, relation_id = batch

            # define a new function to compute loss values for both output_modes
            # s1_vec, s2_vec, o1_vec, o2_vec, \
            s1_vec, s2_vec, o1_vec, o2_vec, loss = model(input_ids, token_type_ids=segment_ids,
                                                         attention_mask=input_mask,
                                                         s1=s1, s2=s2, k1=k1, k2=k2, o1=o1, o2=o2,
                                                         relation_id=relation_id
                                                         )

            print_train_metrics(s1_vec, s2_vec, s1, s2, input_mask,
                                o1_vec=o1_vec, o2_vec=o2_vec, o1=o1, o2=o2)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            # logger.info("loss:{}".format(loss.cpu().item()))
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if fp16:
                # modify learning rate with special warm up BERT uses
                # if args.fp16 is False, BertAdam is used that handles this automatically
                lr_this_step = lr * warmup_linear.get_lr(global_step, warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if (fp16):
                tb_writer.add_scalar('lr', lr_this_step, global_step)
            else:
                tb_writer.add_scalar('lr', optimizer.get_lr()[0], global_step)
            tb_writer.add_scalar('loss', loss.item(), global_step)

if do_eval:

    # eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    eval_dataloader = create_dataloder(dev_examples)
    model.cuda()
    model.eval()
    eval_loss = 0
    # nb_eval_steps = 0
    # preds = []
    # out_label_ids = None

    for input_ids, input_mask, segment_ids, s1, s2, k1, k2, o1, o2, relation_id in tqdm(eval_dataloader,
                                                                                        desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        s1 = s1.to(device)
        s2 = s2.to(device)
        k1 = k1.to(device)
        k2 = k2.to(device)
        o1 = o1.to(device)
        o2 = o2.to(device)
        relation_id = relation_id.to(device)

        with torch.no_grad():
            s1_vec, s2_vec, o1_vec, o2_vec, loss = model(input_ids, token_type_ids=segment_ids,
                                                         attention_mask=input_mask,
                                                         s1=s1, s2=s2, k1=k1, k2=k2, o1=o1, o2=o2,
                                                         relation_id=relation_id
                                                         )

        # # create eval loss and other metric required by the task
        # if output_mode == "classification":
        #     loss_fct = CrossEntropyLoss(weight=torch.Tensor([0.5, 1]).cuda())
        #     tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        # elif output_mode == "regression":
        #     loss_fct = MSELoss()
        #     tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        # eval_loss += tmp_eval_loss.mean().item()
        # nb_eval_steps += 1
        # if len(preds) == 0:
        #     preds.append(logits.detach().cpu().numpy())
        #     out_label_ids = label_ids.detach().cpu().numpy()
        # else:
        #     preds[0] = np.append(
        #         preds[0], logits.detach().cpu().numpy(), axis=0)
        #     out_label_ids = np.append(
        #         out_label_ids, label_ids.detach().cpu().numpy(), axis=0)

    # result = {}
    # eval_loss = eval_loss
    # preds = preds[0]
    #
    # loss = tr_loss / global_step
    #
    # result['eval_loss'] = eval_loss
    # result['global_step'] = global_step
    # result['loss'] = loss
    #
    # output_eval_file = os.path.join("output", "eval_results.txt")
    # with open(output_eval_file, "w") as writer:
    #     logger.info("***** Eval results *****")
    #     for key in sorted(result.keys()):
    #         logger.info("  %s = %s", key, str(result[key]))
    #         writer.write("%s = %s\n" % (key, str(result[key])))

if do_inference:
    text = ""
