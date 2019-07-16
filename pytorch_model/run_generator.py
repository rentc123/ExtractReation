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
import random
from pytorch_model.Bert_model_generator import MyBert
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
do_inference = False
shuffle = True
lr = 5e-5
batch_size = 32
num_train_epochs = 5
fp16 = True
loss_scale = 0
warmup_proportion = 0.1
every_steps_save = 50


VACB_PATH = "/home/rentc/project/bert_chinese_model/bert-base-chinese-vocab.txt"
BERT_PRETRAINED_PATH = "/home/rentc/project/bert_chinese_model/download_bert-base-chinese.tar.gz"
data_dir = "/home/rentc/project/比赛/ccks2019人物关系抽取/data/after_process/"
Relation_ID_PATH = "/home/rentc/project/比赛/ccks2019人物关系抽取/data/after_process/relation2id.csv"
output_model_file = "output/ccks-rwgx-0709-half.model"



id2relation, relation2id = {}, {}
id2relation, relation2id = init_relation(Relation_ID_PATH)
num_class = len(id2relation)
token_dict = init_token_dict(VACB_PATH)
tokenizer = MyTokenizer(token_dict)
no_cuda = False

model = MyBert.from_pretrained(BERT_PRETRAINED_PATH, num_class=num_class, max_len=MAX_LEN)
if not fp16:
    model = MyBert.from_pretrained(BERT_PRETRAINED_PATH, num_class=num_class, max_len=MAX_LEN, use_fp16=False)

if (os.path.exists(output_model_file)):
    model.load_state_dict(torch.load(output_model_file))

device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")


n_gpu = torch.cuda.device_count()
model.to(device)
if (fp16):
    model.half()


def print_train_metrics(s1_vec, s2_vec, s1, s2, input_mask,
                        o1_vec=None, o2_vec=None, o1=None, o2=None, K1=None, K2=None, use_fp16=True):
    from sklearn.metrics import f1_score
    if (use_fp16):
        mask = input_mask.float().half()
    else:
        mask = input_mask.float()
    s1_p = (s1_vec * mask).cpu().detach().numpy()
    s1 = s1.cpu().detach().numpy()

    s2_p = (s2_vec * mask).cpu().detach().numpy()
    s2 = s2.cpu().detach().numpy()

    o1_p = (o1_vec * mask.view(-1, 1, MAX_LEN)).cpu().detach().numpy()
    o1 = o1.cpu().detach().numpy()

    o2_p = (o2_vec * mask.view(-1, 1, MAX_LEN)).cpu().detach().numpy()
    o2 = o2.cpu().detach().numpy()
    K1 = K1.cpu().detach().numpy()
    K2 = K2.cpu().detach().numpy()
    r = o1.sum(axis=2)  # shape : batch_size * num_class
    actual_relation_ids = r.argmax(-1)
    f1score_sbj = 0
    f1score_obj = 0
    correct = 0
    for i in range(s1_p.shape[0]):
        class_id = actual_relation_ids[i]
        pre1 = np.zeros_like(s1[i])
        pre1[s1_p[i] > 0.5] = 1
        y1 = s1[i]
        pre2 = np.zeros_like(s2[i])
        pre2[s2_p[i] > 0.5] = 1
        y2 = s2[i]
        f1_s1 = f1_score(pre1, y1)
        f1_s2 = f1_score(pre2, y2)
        f1score_sbj += (f1_s1 + f1_s2) / 2

        t1 = np.where(pre1 == 1)[0]
        t2 = np.where(pre2 == 1)[0]
        sbj_list = []
        for k1 in t1:
            k2 = t2[t2 > k1]
            if (len(k2) > 0):
                sbj_list.append((k1, k2[0]))

        y1 = o1[i][class_id]
        y2 = o2[i][class_id]
        p1 = o1_p[i][class_id]
        pre1 = np.zeros_like(y1)
        pre1[p1 > 0.5] = 1
        f1_o1 = f1_score(pre1, y1)
        p2 = o2_p[i][class_id]
        pre2 = np.zeros_like(y2)
        pre2[p2 > 0.5] = 1
        f1_o2 = f1_score(pre2, y2)

        f1score_obj += (f1_o1 + f1_o2) / 2

        O1 = np.argmax(y1)
        O2 = np.argmax(y2)

        Y = (K1[i], K2[i], class_id, O1, O2)
        obj_list = []
        for rid in range(num_class):
            t1 = o1_p[i][rid]
            q1 = np.where(t1 > 0.4)[0]
            t2 = o2_p[i][rid]
            q2 = np.where(t2 > 0.4)[0]
            if (len(q1) > 0 and len(q2) > 0):
                for qx1 in q1:
                    qx2 = q2[q2 > qx1]
                    if (len(qx2) > 0):
                        obj_list.append((rid, qx1, qx2[0]))

                # print(class_id, "-----", rid, "---->", (f1_o1, f1_o2))

        for sbj in sbj_list:
            for obj in obj_list:
                # print(Y, "--->", (sbj[0], sbj[1], obj[0], obj[1], obj[2]))
                if ((sbj[0], sbj[1], obj[0], obj[1], obj[2]) == Y):
                    correct += 1

    logger.info("subject_avg_f1:{:.3f}, f1score_obj:{:.3f},correct_percent:{:.3f}".format
                (f1score_sbj / s1_p.shape[0],
                 f1score_obj / s1_p.shape[0],
                 correct / s1_p.shape[0]))
    # logger.info("obj f1_score:{}".format(f1score_obj / s1_p.shape[0]))
    # logger.info("f1_score:{},correct_percent:{}".format(f1 / s1_p.shape[0], correct / s1_p.shape[0]))


def extract_item(text, use_rid=False):
    text = text[:MAX_LEN - 2]
    input_id, token_type_id = tokenizer.encode(first=text, max_len=MAX_LEN)
    attention_mask = [1] * (len(text) + 2)
    attention_mask += [0] * (MAX_LEN - len(attention_mask))

    input_ids = torch.tensor([input_id], dtype=torch.long).to(device)
    token_type_ids = torch.tensor([token_type_id], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)

    s1_p, s2_p = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    if (fp16):
        attention_mask = attention_mask.half()
    else:
        attention_mask = attention_mask.float()
    s1_p = (s1_p * attention_mask).cpu().detach().numpy()
    s2_p = (s2_p * attention_mask).cpu().detach().numpy()
    i = 0
    pre1 = np.zeros(MAX_LEN)
    pre1[s1_p[i] > 0.4] = 1
    pre2 = np.zeros(MAX_LEN)
    pre2[s2_p[i] > 0.4] = 1
    # print(pre1,pre2)

    t1 = np.where(pre1 == 1)[0]
    t2 = np.where(pre2 == 1)[0]
    sbj_list = []
    for k1 in t1:
        k2 = t2[t2 > k1]
        if (len(k2) > 0):
            sbj_list.append((k1, k2[0]))
    spo_list = []
    for sbj in sbj_list:
        k1, k2 = sbj
        k1_ = torch.tensor([k1], dtype=torch.long).to(device)
        k2_ = torch.tensor([k2], dtype=torch.long).to(device)
        o1_p, o2_p = model(input_ids, token_type_ids, attention_mask, k1=k1_, k2=k2_)

        o1_p = (o1_p * attention_mask.view(-1, 1, MAX_LEN)).cpu().detach().numpy()
        o2_p = (o2_p * attention_mask.view(-1, 1, MAX_LEN)).cpu().detach().numpy()

        for rid in range(num_class):
            t1 = o1_p[i][rid]
            q1 = np.where(t1 > 0.09)[0]
            t2 = o2_p[i][rid]
            q2 = np.where(t2 > 0.09)[0]
            if (len(q1) > 0 and len(q2) > 0):
                for qx1 in q1:
                    qx2 = q2[q2 > qx1]
                    if (len(qx2) > 0):
                        if (use_rid):
                            spo_list.append((text[k1 - 1:k2], rid, text[qx1 - 1: qx2[0]]))
                        else:
                            spo_list.append((text[k1 - 1:k2], id2relation[rid], text[qx1 - 1: qx2[0]]))
    # print(spo_list)
    return spo_list


class data_generator:
    def __init__(self, data, batch_size=32, MaxLen=MAX_LEN, epoch=3):
        self.data = data
        self.MaxLen = MaxLen
        self.batch_size = batch_size
        self.epoch = epoch
        self.len = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.len += 1

    def __len__(self):
        return self.len

    def __iter__(self):
        idxs = list(range(len(self.data)))
        np.random.shuffle(idxs)
        I, T, A, S1_vec, S2_vec, K1, K2, Obj1_vec, Obj2_vec = [], [], [], [], [], [], [], [], []
        for i in idxs:
            example = self.data[i]

            text = example["text"][:MAX_LEN - 2]
            input_id, token_type_id = tokenizer.encode(first=text, max_len=MAX_LEN)
            attention_mask = [1] * (len(text) + 2)
            attention_mask += [0] * (MAX_LEN - len(attention_mask))
            sbj1_vec = np.zeros(MAX_LEN)
            sbj2_vec = np.zeros(MAX_LEN)
            obj1_vec = np.zeros((num_class, MAX_LEN))
            obj2_vec = np.zeros((num_class, MAX_LEN))

            item_list = []  # (s1,s2)-->(relation_id,o1,o2)

            for spo in example["spo_list"]:
                sbj = spo["subject"]
                obj = spo["object"]
                relation_id = (int)(spo["relation_id"])
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
                    sbj1_vec[s1] = 1
                    sbj2_vec[s2] = 1
                    item_list.append([s1, s2, relation_id, o1, o2])

            if (item_list):
                indx = random.randint(0, len(item_list) - 1)
                k1 = item_list[indx][0]
                k2 = item_list[indx][1]
                relation_id = item_list[indx][2]
                o1 = item_list[indx][3]
                o2 = item_list[indx][4]
                obj1_vec[relation_id][o1] = 1
                obj2_vec[relation_id][o2] = 1
            I.append(input_id)
            T.append(token_type_id)
            A.append(attention_mask)
            S1_vec.append(sbj1_vec)
            S2_vec.append(sbj2_vec)
            K1.append(k1)
            K2.append(k2)
            Obj1_vec.append(obj1_vec)
            Obj2_vec.append(obj2_vec)

            if (len(I) == self.batch_size or i == idxs[-1]):
                yield [I, T, A, S1_vec, S2_vec, K1, K2, Obj1_vec, Obj2_vec]
                I, T, A, S1_vec, S2_vec, K1, K2, Obj1_vec, Obj2_vec = [], [], [], [], [], [], [], [], []


if do_train:

    data = json.load(open(os.path.join(data_dir, "dev.json"), 'r'))
    data = data[0:10000]
    train_data = json.load(open(os.path.join(data_dir, "train.json"), 'r'))
    # train_data = train_data[0:]
    data.extend(train_data)

    train_generator = data_generator(data)
    tb_writer = SummaryWriter()
    model.train()

    num_train_optimization_steps = (len(data) // batch_size + 1) * num_train_epochs

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
        warmup_linear = WarmupLinearSchedule(warmup=warmup_proportion,
                                             t_total=num_train_optimization_steps)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=lr,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(data))
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    model.train()
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    for _ in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, example in enumerate(train_generator):

            batch = tuple(torch.tensor(t, dtype=torch.long).to(device) for t in example)
            input_ids, segment_ids, input_mask, s1, s2, k1, k2, o1, o2 = batch

            s1_vec, s2_vec, o1_vec, o2_vec, loss = model(input_ids, token_type_ids=segment_ids,
                                                         attention_mask=input_mask,
                                                         s1=s1, s2=s2, k1=k1, k2=k2, o1=o1, o2=o2
                                                         )

            print_train_metrics(s1_vec, s2_vec, s1, s2, input_mask,
                                o1_vec=o1_vec, o2_vec=o2_vec, o1=o1, o2=o2, K1=k1, K2=k2, use_fp16=fp16)

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
            logger.info("epoch:{:d},step:{:d}/{:d},loss:{:.3f}".format(_, step,
                                                                       num_train_optimization_steps // num_train_epochs,
                                                                       loss.cpu().detach().item()))
            if (global_step % every_steps_save == 0):
                torch.save(model.state_dict(), output_model_file)

if do_eval:
    data = json.load(open(os.path.join(data_dir, "dev.json"), 'r'))
    data = data[10000:]
    pre_all = 0
    actual_all = 0
    correct = 0
    for i, example in enumerate(tqdm(data)):
        text = example["text"]
        spo_list = example["spo_list"]
        pre_spo_list = extract_item(text, use_rid=True)
        pre_all += len(pre_spo_list)
        actual_spo_list = []
        for spo in spo_list:
            k = (spo["subject"], spo["relation_id"], spo["object"])
            actual_all += 1
            if (k in pre_spo_list):
                correct += 1
        precise = correct / pre_all
        recall = correct / actual_all
        f1 = 2 * precise * recall / (precise + recall)
        if (i % 20 == 0):
            print("precise:{:.3f},recall:{:.3f},f1_score:{:.3f}".format(precise, recall, f1))
    #
    # output_eval_file = os.path.join("output", "eval_results.txt")
    # with open(output_eval_file, "w") as writer:
    #     logger.info("***** Eval results *****")
    #     for key in sorted(result.keys()):
    #         logger.info("  %s = %s", key, str(result[key]))
    #         writer.write("%s = %s\n" % (key, str(result[key])))

if do_inference:
    # text = "周杰伦唱了一首歌名字叫《轨迹》。"
    # text = "张书文，男，1964年3月出生，汉族，湖北随州广水市人，在职研究生，会计师，审计师，民建会员，1982年8月参加工作"
    text = "《大明按察使》该剧由黄海涛，申积军，阙云霞，沈剑波，王江红担任策划、黄克敏执导，姚橹，丁勇岱，杨旸，高鑫，李芯逸等主演的悬疑断案电视剧"
    text = "2010年06月30日21时25分许,王红发(男、1982年03月18日生、身份证:371321198203181438、山东省沂南县岸堤镇辛兴村二组5号)报警:长沙市芙蓉区邵阳坪1号402房有纠纷。"
    text = "《战狼》、《战狼2》的主演都是吴京"
    text = "《战狼》的主演和导演都是吴京"
    text = "《鲁迅自传》由江苏文艺出版社出版"
    print(extract_item(text))
