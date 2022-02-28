import argparse
import numpy as np
import random
import time
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm, trange
# import dgl
# from dgl.data import register_data_args, load_data
# from gat_merge_cls import GAT
from sklearn.metrics import f1_score
from optimization import WarmupLinearSchedule, AdamW
# from nltk.corpus import wordnet
# from nltk import word_tokenize, pos_tag
# from nltk.stem import WordNetLemmatizer
# import h5py
import gc
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizer
from enum import Enum
# from allennlp.modules.elmo import Elmo, batch_to_ids
# from rgcn_cls import RGCN
import os
import torch.nn as nn
import requests
import sys
import copy
import datetime
import time

from models.GCN import GCN

EMBEDDING_FILE = 'data/glove.840B.300d.txt'

vocab_size = 0
for i, line in enumerate(open(EMBEDDING_FILE, encoding='utf-8')):
    vocab_size += 1

embedding_matrix = np.zeros((vocab_size + 1, 300))
word2idx = {}

for i, line in enumerate(open(EMBEDDING_FILE, encoding='utf-8')):
    val = line.split()
    word2idx["".join(val[:-300])] = i + 1
    embedding_matrix[i + 1] = np.asarray(val[-300:], dtype='float32')

word2idx["<UNK>"] = 0

embedding_matrix = torch.FloatTensor(embedding_matrix)

def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).to(dtype=torch.bool), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result

def word2index(sent):
    s = []
    for token in sent:
        try:
            s.append(word2idx[token])
        except:
            s.append(0)
    return s

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    adj, sent1, sent2, labels = map(list, zip(*samples))
    # batched_graph = dgl.batch(graphs)
    return adj, sent1, sent2, torch.LongTensor(labels)


def send_graph_to_device(g, device):
    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device)

    # edges
    labels = g.edge_attr_schemes()
    for l in labels.keys():
        g.edata[l] = g.edata.pop(l).to(device)

    return g


def load_depen(data_dir, dependency_results):
    with open(data_dir) as fr:
        dependency_result = []
        for line in fr.readlines():
            if line == '\n':
                if dependency_result == []:
                    dependency_result.append(['root', '_ROOT', '0', 'None', '1'])
                    # dependency_results.append([['root', '_ROOT', '0', '', '1']])
                    # dependency_result = []
                else:
                    dependency_results.append(dependency_result)
                    dependency_result = []
            else:
                line = line.strip('\n').split('\t')
                dependency_result.append(line)


def dependency_results_process(srcs, dsts, sentences, result):
    sent = []
    src = []
    dst = []
    for res in result:
        sent.append(res[3])
        src.append(int(res[2]))
        try:
            dst.append(int(res[4]))
        except IndexError:
            print(result)
            exit()
    sentences.append(sent)
    srcs.append(src)
    dsts.append(dst)
    # all_sent_len.append(dst[-1])


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    res = []
    lemmatizer = WordNetLemmatizer()
    for word, pos in pos_tag(sentence):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))

    return res


def find_same_token(sent1, sent2, sent1_len_max):
    src = []
    dst = []
    try:
        sent1 = lemmatize_sentence(sent1)
        sent2 = lemmatize_sentence(sent2)
    except IndexError:
        print(sent1)
        print(sent2)
        exit()
    len1 = len(sent1)
    len2 = len(sent2)

    for i in range(len1):
        # judgement of stopwords i
        for j in range(len2):
            # judgement of stopwords j
            if sent1[i] == sent2[j]:
                src.append(i + 1)
                dst.append(j + 1 + sent1_len_max)

    return src, dst


class GDataset(object):
    def __init__(self, Gset, Gsent, Glabels):
        super(GDataset, self).__init__()
        self.Gset = Gset
        # self.Gcls = Gcls
        self.Gsent = Gsent
        self.Glabels = Glabels

    def __getitem__(self, idx):
        # return self.Gset[idx], self.Gcls[idx], self.Glabels[idx]
        return self.Gset[idx], self.Gsent[idx*2], self.Gsent[idx*2+1], self.Glabels[idx]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.Gset)


class Merge_method(Enum):
    mean = 1
    first = 2


def _truncate_q_pair(tokens_q1, tokens_q2, max_total_seq_length):
    len_spare = len(tokens_q1) + len(tokens_q2) - max_total_seq_length
    is_truncated = len_spare > 0
    # print("len_spare：%d" %len_spare)

    while len_spare > 0:
        is_truncated = True
        if len(tokens_q1) >= len(tokens_q2):
            tokens_q1.pop()
        else:
            tokens_q2.pop()
        len_spare -= 1

    return is_truncated


def _find_corr_features(output_features, list_original, list_trancated, merge_type, encoder_type):
    curr_pos = 0
    token_target = ""
    idx_list = []
    feature_tensor = []

    if encoder_type == 'bert':
        flag = True

        for idx, token in enumerate(list_trancated):
            if token == list_original[curr_pos].lower() \
                        or (len(token) == len(list_original[curr_pos].lower()) and flag == True) \
                        or token == "[UNK]":
                curr_pos += 1
                idx_list.append([idx])
                token_target = ""
                continue
            elif token.startswith('##'):
                token_target += token.lstrip('#')
                idx_list[-1].append(idx)
            else:
                token_target += token
                if flag:
                    idx_list.append([idx])
                    flag = False
                else:
                    idx_list[-1].append([idx])

            if token_target == list_original[curr_pos].lower() or len(token_target) == len(list_original[curr_pos].lower()):
                curr_pos += 1
                token_target = ""
                flag = True
    elif encoder_type == 'roberta':
        # ['Even', 'Ġif', 'Ġhe', 'Ġis', 'Ġsorting', 'Ġthe', 'Ġaddresses', 'Ġby', 'Ġhand', 'Ġ,', 'Ġhe',
        # 'Ġhas', 'Ġthe', 'Ġoption', 'Ġof', 'Ġdoing', 'Ġthe', 'Ġwork', 'Ġin', 'Ġa', 'Ġcompletely',
        # 'Ġdifferent', 'Ġway', 'ĠAlso', 'Ġ,', 'Ġif', 'Ġthe', 'Ġsame', 'Ġmailing', 'Ġlist', 'Ġis',
        # 'Ġused', 'Ġmore', 'Ġthan', 'Ġonce', 'Ġ,', 'Ġor', 'Ġis', 'Ġused', 'Ġagain', 'Ġwith', 'Ġslight',
        # 'Ġmodification', 'Ġ,', 'Ġhe', 'Ġcan', 'Ġsort', 'Ġonce', 'Ġand', 'Ġdo', 'Ġmany', 'Ġmail',
        # 'ings', 'Ġ.']
        for idx, token in enumerate(list_trancated):
            if not idx_list or list_trancated[idx].startswith("\u0120"):
                idx_list.append([idx])
            else:
                idx_list[-1].append(idx)
    else:
        exit(1)

    assert merge_type == 'mean' or merge_type == 'first'
    for sub_idx_list in idx_list:
        if merge_type == 'mean':
            sub_feature = torch.mean(output_features[:, sub_idx_list[:], :], dim=1, keepdim=False)
        else:
            sub_feature = output_features[:, sub_idx_list[0], :]

        sub_feature = sub_feature.unsqueeze(dim=1)
        if len(feature_tensor) > 0:
            feature_tensor = torch.cat((feature_tensor, sub_feature), 1)
        else:
            feature_tensor = sub_feature

    return feature_tensor


def _restore_features(output_features, target_list1, target_list2, tokens, merge_type, encoder_type):
    cls_feature = output_features[:, 0, :]

    sep1_pos = tokens.index("[SEP]")
    text1_feature = _find_corr_features(output_features[:, 1:sep1_pos, :], target_list1, tokens[1:sep1_pos],
                                        merge_type, encoder_type)
    text2_feature = _find_corr_features(output_features[:, sep1_pos+1:-1, :], target_list2, tokens[sep1_pos+1:-1],
                                        merge_type, encoder_type)

    return text1_feature, text2_feature, cls_feature


def gen_bert_feature(bert_tokenizer, bert_model, device, text1, text2, target_list1, target_list2, max_seq_length,
                     merge_type, encoder_type):
    assert len(text1) > 0 and len(text2) > 0

    tokens1 = bert_tokenizer.tokenize(text1)
    tokens2 = bert_tokenizer.tokenize(text2)

    is_truncated = _truncate_q_pair(tokens1, tokens2, max_seq_length - 3)

    tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]
    segment_ids = [0] * (len(tokens1) + 2) + [1] * (len(tokens2) + 1)

    input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([input_ids])
    segments_tensors = torch.tensor([segment_ids])

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)

    # Predict hidden states features for each layer
    # output_features, _ = bert_model(input_ids=tokens_tensor, token_type_ids=segments_tensors, output_all_encoded_layers=False)
    
    # package transformers is different from package pytorch_pretrained_bert
    tensors = bert_model(input_ids=tokens_tensor, token_type_ids=segments_tensors)
    output_features = tensors[0]
    

    # bert_model返回值：
    # last_hidden_state：shape是(batch_size, sequence_length, hidden_size)，hidden_size=768,它是模型最后一层输出的隐藏状态
    # pooler_output：shape是(batch_size, hidden_size)，这是序列的第一个token(classification token)的最后一层的隐藏状态，它是由线性层和Tanh激活函数进一步处理的，这个输出不是对输入的语义内容的一个很好的总结，对于整个输入序列的隐藏状态序列的平均化或池化通常更好。
    # hidden_states：这是输出的一个可选项，如果输出，需要指定config.output_hidden_states=True,它也是一个元组，它的第一个元素是embedding，其余元素是各层的输出，每个元素的形状是(batch_size, sequence_length, hidden_size)
    # attentions：这也是输出的一个可选项，如果输出，需要指定config.output_attentions=True,它也是一个元组，它的元素是每一层的注意力权重，用于计算self-attention heads的加权平均值
    # 总结：所以通常来说，只输出前两项
    # ————————————————
    # 版权声明：本文为CSDN博主「乐清sss」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
    # 原文链接：https://blog.csdn.net/sunyueqinghit/article/details/105157609

    # output_features.to(device)
    tokens1_feature, tokens2_feature, cls_feature = \
        _restore_features(output_features, target_list1, target_list2, tokens, merge_type, encoder_type)

    return cls_feature, tokens1_feature[0], tokens2_feature[0]


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {"acc": simple_accuracy(preds, labels)}
    # return acc_and_f1(preds, labels)


def fetch_data_and_graph(hyps, force_re_parsing_depen=False):
    # dependency_file_list, label_file_list, edge_types, finetune_depen, use_glob_node,

    if hyps["eval_on_hans"]:
        dependency_file_list = [hyps["eval_data_list"]]
    else:
        dependency_file_list = hyps["input_data_list"]
        label_file_list = hyps["input_label_list"]
        assert len(dependency_file_list) == len(label_file_list)

    use_gnn = hyps["use_gnn"]
    if use_gnn:
        edge_types = hyps["gnn"]["edge_types"]
    else:
        edge_types = 4 # no matter whether using gnn, the data file is created as the same.

    sentences = []
    labels = []
    adj = []

    # G = []
    # dependency_results = []
    # srcs = []
    # dsts = []
    # all_sent_len = []

    file_name = "mnli_all_data.pt"
    # file_name = "mnli_all_data_" + hyps["mutual_link"] + ".pt"

    # for file in dependency_file_list:
    #     file_name += "".join(file)

    # if finetune_depen:
    #     file_name_all += "finetunning parsing"
    # else:
    #     file_name_all += "with existing parsing result"

    # if use_glob_node:
    #     file_name_all += "with global node"
    # else:
    #     file_name_all += "without global node"

    # file_name_all += "same coattn"

    # file_name_all_coded = fetch_file_name_coded(file_name_all) + ".pt"
    file_name_all = os.path.join("data/graph_input", file_name)

    if not hyps["eval_on_hans"] and not force_re_parsing_depen and os.path.exists(file_name_all):
        print("Loading graph from existing file %s..." % file_name_all)
        sentences, adj, labels, len_list = torch.load(file_name_all)
        # if hyps["mutual_link"] == "same_word":
        #     adj = link_same_word()

        return sentences, adj, labels, len_list

    print("Analysing dependency file...")

    len_list = []

    for idx in tqdm(range(len(dependency_file_list)), desc="Dependency files"):
        dependency_results = []
        srcs = []
        dsts = []

        load_depen(dependency_file_list[idx], dependency_results)
        if hyps["eval_on_hans"]:
            labels_curr = np.zeros(int(len(dependency_results) / 2))
        else:
            labels_curr = np.loadtxt(label_file_list[idx], delimiter=',')
        labels = np.concatenate((labels, labels_curr))

        len_list.append(int(len(dependency_results) / 2))

        # len_accumulated = len(adj)
        # len_accumulated = len(G)

        # assert len(G) == len(labels)

        for i in range(int(len(dependency_results) / 2)):
            dependency_results_process(srcs, dsts, sentences, dependency_results[i * 2])
            dependency_results_process(srcs, dsts, sentences, dependency_results[i * 2 + 1])

        # use external dependency parsing result to obtain dependency relations,
        # or only use the variable "sentences" to achieve dense dependency graph.

        # the ordering of sentence pair: [global node] + [root_sent1] + [sent1]*sent1_len + [root_sent2] + [sent2]*sent2_len
        # the root node of each sentence is added in case that multiple sentences exist in sent1 or sent2.
        # for i in tqdm(range(int(len(dsts) / 2))):
        for i in trange(int(len(dsts) / 2), desc="Iteration"):  # trange(i) equlvalent to tqdm(range(i))
            # g = dgl.DGLGraph()
            len1 = dsts[i*2][-1]  # length of sentence 1
            len2 = dsts[i*2+1][-1]  # length of sentence 2

            # adj_curr = torch.zeros(edge_types, len1+len2+2, len1+len2+2)
            adj_curr1 = torch.zeros(edge_types, len1+1, len1+1)
            adj_curr2 = torch.zeros(edge_types, len2+1, len2+1)

            # create nodes in each sentence pair
            # g.add_nodes(len1 + len2 + 3)

            # create edge from each word to itself in sentence 1 and sentence 2
            for idx in range(len1+1):
                adj_curr1[0][idx][idx] = 1
            for idx in range(len2+1):
                adj_curr2[0][idx][idx] = 1

            # create edge from each word to its dependency head in sentence 1 and sentence 2
            # src = [x + 1 for x in srcs[i * 2]]
            # dst = [x + 1 for x in dsts[i * 2]]
            # g.add_edges(src, dst)
            for idx in range(len1):
                adj_curr1[1][srcs[i*2][idx]][dsts[i*2][idx]] = 1
            for idx in range(len2):
                adj_curr2[1][srcs[i*2+1][idx]][dsts[i*2+1][idx]] = 1

            # create edge from each head to its dependent in sentence 1 and sentence 2
            # src = [x + len1 + 2 for x in srcs[i * 2 + 1]]
            # dst = [x + len1 + 2 for x in dsts[i * 2 + 1]]
            # g.add_edges(src, dst)
            for idx in range(len1):
                adj_curr1[2][dsts[i*2][idx]][srcs[i*2][idx]] = 1
            for idx in range(len2):
                adj_curr2[2][dsts[i*2+1][idx]][srcs[i*2+1][idx]] = 1

            # # create edge from the global node, that is the 1st node, to all the nodes in the two sentences.
            # # src = list(range(1, len1 + len2 + 3))
            # # dst = [0] * (len1 + len2 + 2)
            # # g.add_edges(src, dst)
            # # g.add_edges(dst, src)
            # for idx in range(1, len1+len2+3):
            #     adj_curr[2][idx][0] = 1
            #     adj_curr[2][0][idx] = 1

            # # create edges between the same tokens in sentence pair
            # src, dst = find_same_token(sentences[len_accumulated*2 + i*2], sentences[len_accumulated*2 + i*2 + 1])
            # edge_list = map(list, zip(src, dst))
            # for edge_inst in edge_list:
            #     adj_curr[3][edge_inst[0]][edge_inst[1]] = 1
            #     adj_curr[3][edge_inst[1]][edge_inst[0]] = 1

            adj.append([adj_curr1, adj_curr2])

            # # mask? to be confirmed
            # w = torch.zeros(len1 + len2 + 3, 1)
            # w[0, 0] = 1
            # g.ndata['w'] = w
            # G.append(g)

    assert len(adj) == len(labels)

    if hyps["save_dependency_pt"]:
        torch.save([sentences, adj, labels, len_list], file_name_all)

    return sentences, adj, labels, len_list


class Classifier(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_labels):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(in_dim * 4, num_labels)
        # dropout = 0
        # self.classifier = nn.Sequential(nn.Linear(in_dim * 4, hid_dim), nn.LeakyReLU(0.2),
        # nn.Linear(hid_dim, num_labels))

    def forward(self, data):
        return self.classifier(data)


class Attn(nn.Module):
    def __init__(self, out_size, attn_size):
        super(Attn, self).__init__()

        self.W = nn.Parameter(torch.Tensor(out_size, out_size))

        self.Wv = nn.Parameter(torch.Tensor(out_size, attn_size))
        self.Wq = nn.Parameter(torch.Tensor(out_size, attn_size))

        nn.init.xavier_uniform_(self.Wv, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.Wq, gain=nn.init.calculate_gain('tanh'))
        
        self.w_hv = nn.Parameter(torch.Tensor(attn_size, 1))
        self.w_hq = nn.Parameter(torch.Tensor(attn_size, 1))
        nn.init.xavier_uniform_(self.w_hv, gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.w_hq, gain=nn.init.calculate_gain('linear'))

    def forward(self, seq_features1, seq_features2, mask1, mask2):
        C = F.tanh(torch.matmul(torch.matmul(seq_features1, self.W), torch.transpose(seq_features2, 1, 2)))

        Hv = F.tanh(torch.matmul(seq_features1, self.Wv) + torch.matmul(C, torch.matmul(seq_features2, self.Wq)))
        Hq = F.tanh(torch.matmul(seq_features2, self.Wq) + torch.matmul(torch.transpose(C, 1, 2), torch.matmul(seq_features1, self.Wv)))

        attn_v = masked_softmax(torch.matmul(Hv, self.w_hv).squeeze(), mask1, 1)
        attn_q = masked_softmax(torch.matmul(Hq, self.w_hq).squeeze(), mask2, 1)

        v_hat = torch.sum(torch.unsqueeze(attn_v, 2) * seq_features1, 1)
        q_hat = torch.sum(torch.unsqueeze(attn_q, 2) * seq_features2, 1)

        return v_hat, q_hat


class MatchNetwork(nn.Module):
    def __init__(self, hyps, t_total):
        super(MatchNetwork, self).__init__()

        self.device = hyps["device"]
        self.dropout_inter = nn.Dropout2d(p=hyps["dropout_inter"])
        self.num_labels = hyps["num_labels"]

        self.depen_network = None
        self.embed_size = hyps["embed_size"]

        self.bert_model = None
        self.rnn = None
        self.encoder_type = hyps["encoder_type"]
        if hyps["encoder_type"] == "bert":
            print("Initializing BERT model...")
            bert_model_name = 'bert-base-uncased'
            self.max_seq_length = 512

            # merge_type should be 'mean' or 'first'
            self.merge_type = "first"
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.bert_model = BertModel.from_pretrained(bert_model_name)
            self.bert_model.to(hyps["device"])
            self.bert_model.zero_grad()
        elif hyps["encoder_type"] == "roberta":
            print("Initializing RoBERTa model...")
            roberta_model_name = 'roberta-base'
            self.max_seq_length = 512

            # merge_type should be 'mean' or 'first'
            self.merge_type = "first"
            self.bert_tokenizer = RobertaTokenizer.from_pretrained(roberta_model_name)
            self.bert_model = RobertaModel.from_pretrained(roberta_model_name)
            self.bert_model.to(hyps["device"])
            self.bert_model.zero_grad()
        elif hyps["encoder_type"] == "lstm":
            self.rnn = nn.LSTM(hyps["embed_size"], int(hyps["encoder_output_size"]/2), 2, batch_first=True,
                               bidirectional=True)
            self.rnn.to(hyps["device"])
        else:
            print("error encoder type!")
            exit(1)

        self.GCNModel = None
        if hyps["use_gnn"] == True:
            self.GCNModel = GCN(hyps["gnn"], hyps["mutual_link"], self.device)
            self.mutual_link = hyps["mutual_link"]

        if self.GCNModel:
            self.attn = Attn(hyps["gnn"]["out_features"], hyps["attn_size"])
            self.classifier = Classifier(hyps["gnn"]["out_features"], hyps["classifier_hid_dim"], self.num_labels)
        else:
            self.attn = Attn(hyps["encoder_output_size"], hyps["attn_size"])
            self.classifier = Classifier(hyps["encoder_output_size"], hyps["classifier_hid_dim"], self.num_labels)

        if self.bert_model:
            self.optimizer_bert = AdamW(self.bert_model.parameters(),
                                        lr=3e-5,
                                        weight_decay=hyps["weight_decay"],
                                        eps=hyps["adam_epsilon"])
            self.scheduler_bert = WarmupLinearSchedule(self.optimizer_bert,
                                                       warmup_steps=hyps["warmup_steps"],
                                                       t_total=t_total)

        other_para = []
        if self.rnn:
            other_para.append({'params': self.rnn.parameters()})
        if self.GCNModel:
            other_para.append({'params': self.GCNModel.parameters()})
        other_para.append({'params': self.attn.parameters()})
        other_para.append({'params': self.classifier.parameters()})

        self.optimizer_other = torch.optim.Adam(other_para,
                                                lr=hyps["lr"],
                                                weight_decay=hyps["weight_decay"],
                                                eps=hyps["adam_epsilon"])

        self.loss_fct = nn.CrossEntropyLoss()

        self.scheduler_other = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_other, milestones=[2], gamma=0.1, verbose=True)

        self.to_device(hyps["device"])

        self.GCNModel.zero_grad()
        self.bert_model.zero_grad()

    def to_device(self, device):
        if self.GCNModel:
            self.GCNModel.to(device)
        if self.rnn:
            self.rnn.to(device)
        if self.bert_model:
            self.bert_model.to(device)
        self.classifier.to(device)
        self.attn.to(device)


    def to_train(self, device):
        if self.GCNModel:
            self.GCNModel.train()
        if self.rnn:
            self.rnn.train()
        if self.bert_model:
            self.bert_model.train()
        self.attn.train()
        self.classifier.train()


    def forward(self, data_batch):
        adj_list, sent1, sent2, label = data_batch
        # ubg = dgl.unbatch(bg)

        # clss = torch.Tensor([]).to(self.device)
        # embed_list = []
        embed1_batch = torch.Tensor([]).to(self.device)
        embed2_batch = torch.Tensor([]).to(self.device)
        mask1_batch = torch.Tensor([]).to(self.device)
        mask2_batch = torch.Tensor([]).to(self.device)
        adj_batch = torch.Tensor([]).to(self.device)

        root_node = 1 if self.GCNModel else 0

        # if self.bert_model:
        #     embed_size = 768
        # if self.rnn:
        #     embed_size = 300

        # seq1_len = 0
        # for i in range(len(adj_list)):
        #     seq1_len = max(seq1_len, len(sent1[i]) + root_node)
        seq1_len = max([(len(seq1) + root_node) for seq1 in sent1])

        # seq2_len = 0
        # for i in range(len(adj_list)):
        #     seq2_len = max(seq2_len, len(sent2[i]) + root_node)
        seq2_len = max([(len(seq2) + root_node) for seq2 in sent2])

        for i in range(len(adj_list)):
            # print(ubg[i].number_of_nodes())
            # print(sent1[i])
            # print(sent2[i])

            if self.bert_model:
                cls, embed1, embed2 = gen_bert_feature(self.bert_tokenizer, self.bert_model, self.device,
                                                       " ".join(sent1[i]), " ".join(sent2[i]),
                                                       sent1[i], sent2[i],
                                                       self.max_seq_length, self.merge_type, self.encoder_type)

            if self.rnn:
                embed1 = F.embedding(torch.LongTensor(word2index(sent1[i])), embedding_matrix).to(self.device)
                embed2 = F.embedding(torch.LongTensor(word2index(sent2[i])), embedding_matrix).to(self.device)

            assert len(sent1[i]) == len(embed1) and len(sent2[i]) == len(embed2)

            seq1_len_curr = len(embed1) + root_node
            seq2_len_curr = len(embed2) + root_node
            # print("seq_len_curr:%d" % seq_len_curr)
            # assert seq1_len_curr == adj_list[i][0].size(1) and seq2_len_curr == adj_list[i][1].size(1)

            # clss = torch.cat((clss, cls), 0)
            # print(embed1.size())
            # print(embed2.size())
            # ubg[i].ndata["h"] = torch.cat((torch.zeros((2, 768)).to(device), embed1, torch.zeros((1, 768)).to(device), embed2), 0)

            embed1_curr = torch.cat((torch.zeros(root_node, self.embed_size).to(self.device), embed1))
            embed1_curr = torch.cat((embed1_curr, torch.zeros(seq1_len - seq1_len_curr, self.embed_size).to(self.device)), 0)
            embed1_batch = torch.cat((embed1_batch, torch.unsqueeze(embed1_curr, 0)), 0)

            embed2_curr = torch.cat((torch.zeros(root_node, self.embed_size).to(self.device), embed2))
            embed2_curr = torch.cat((embed2_curr, torch.zeros(seq2_len - seq2_len_curr, self.embed_size).to(self.device)), 0)
            embed2_batch = torch.cat((embed2_batch, torch.unsqueeze(embed2_curr, 0)), 0)

            mask1_curr = [1] * seq1_len_curr + [0] * (seq1_len - seq1_len_curr)
            mask1_curr = torch.tensor([mask1_curr], dtype=torch.float).to(self.device)
            mask1_batch = torch.cat((mask1_batch, mask1_curr), 0)

            mask2_curr = [1] * seq2_len_curr + [0] * (seq2_len - seq2_len_curr)
            mask2_curr = torch.tensor([mask2_curr], dtype=torch.float).to(self.device)
            mask2_batch = torch.cat((mask2_batch, mask2_curr), 0)

            if self.GCNModel:
                adj1_curr = copy.deepcopy(adj_list[i][0]).to(self.device)
                adj1_curr = torch.cat((adj1_curr, torch.zeros(adj1_curr.size(0), seq1_len - seq1_len_curr, seq1_len_curr).to(self.device)), 1)
                adj1_curr = torch.cat((adj1_curr, torch.zeros(adj1_curr.size(0), seq1_len, seq1_len + seq2_len - seq1_len_curr).to(self.device)), 2)

                adj2_curr = copy.deepcopy(adj_list[i][1]).to(self.device)
                adj2_curr = torch.cat((adj2_curr, torch.zeros(adj2_curr.size(0), seq2_len - seq2_len_curr, seq2_len_curr).to(self.device)), 1)
                adj2_curr = torch.cat((adj2_curr, torch.zeros(adj2_curr.size(0), seq2_len, seq1_len + seq2_len - seq2_len_curr).to(self.device)), 2)

                adj_curr = torch.cat((adj1_curr, adj2_curr), 1)

                if self.mutual_link == "same_word":
                    # create edges between the same tokens in sentence pair
                    src, dst = find_same_token(sent1[i], sent2[i], seq1_len)
                    edge_list = map(list, zip(src, dst))
                    for edge_inst in edge_list:
                        adj_curr[3][edge_inst[0]][edge_inst[1]] = 1
                        adj_curr[3][edge_inst[1]][edge_inst[0]] = 1

                adj_batch = torch.cat((adj_batch, torch.unsqueeze(adj_curr, 0)), 0)

        # bg = dgl.batch(ubg)
        # bg = send_graph_to_device(bg, device)
        label = label.to(self.device)
        # clss = clss.to(device)
        # print(optimizer)

        embed_combined = self.dropout_inter(torch.cat((embed1_batch, embed2_batch), 1).transpose(1, 2)).transpose(1, 2)
        embed1_batch, embed2_batch = embed_combined[:, :seq1_len, :], embed_combined[:, seq1_len:, :]

        if self.rnn:
            embed1_batch, hidden_state1 = self.rnn(embed1_batch)
            embed2_batch, hidden_state2 = self.rnn(embed2_batch)

        if self.GCNModel:
            embed1_batch, embed2_batch = self.GCNModel(embed1_batch, mask1_batch, embed2_batch, mask2_batch, adj_batch)

        data1, data2 = self.attn(embed1_batch, embed2_batch, mask1_batch, mask2_batch)

        logit = torch.cat((data1, data2, torch.abs(data1 - data2), torch.mul(data1, data2)), 1)

        logit = self.classifier(logit)

        return logit, label

    def loss(self, data_batch):
        logit, label = self.forward(data_batch)
        loss = self.loss_fct(logit.view(-1, self.num_labels), label.view(-1))
        return loss

def evaluate(network, G, sentences, labels, len_list, hyps, result_rec_folder, text=None):
    batch_size = hyps["batch_size"]
    device = hyps["device"]
    preds = None
    out_label_ids = None
    # eval_loss = 0

    if hyps["eval_on_hans"]:
        LABEL_MAP = ["entailment", "neutral", "contradiction"]
        evalset = GDataset(G[:len_list[0]], sentences[:len_list[0] * 2], labels[:len_list[0]])
    else:
        evalset = GDataset(G[len_list[0]:len_list[0] + len_list[1]],
                           sentences[len_list[0] * 2:(len_list[0] + len_list[1]) * 2],
                           labels[len_list[0]:len_list[0] + len_list[1]])
    evaldata_loader = DataLoader(evalset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    for iter, data_batch in enumerate(evaldata_loader):
        network.eval()
        with torch.no_grad():
            logit, label = network(data_batch)
            # loss = network.loss_fct(logit.view(-1, network.num_labels), label.view(-1))
            # eval_loss += loss.detach().item()
            if preds is None:
                preds = logit.detach().cpu().numpy()
                out_label_ids = label.detach().cpu().numpy()
            else:
                preds = np.append(preds, logit.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, label.detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)

    if hyps["eval_on_hans"]:
        print("Writing hans result " + hyps["eval_hans_result"] +" ...")
        F = open(hyps["eval_hans_result"], "w")
        F.write("pairID,gold_label\n")

        sum = 0
        for k in range(len(preds)):
            F.write("ex" + str(sum) + "," + LABEL_MAP[int(preds[k])] + "\n")
            sum += 1

        F.close()
        return

    result = compute_metrics(preds, out_label_ids)

    for key in sorted(result.keys()):
        print("matched %s = %s\n" % (key, str(result[key])))

    if result_rec_folder != None:
        F = open(os.path.join(result_rec_folder, ("matched_results_print" + text + ".txt")), "w")
        for key in sorted(result.keys()):
            F.write(key + " = " + str(result[key]) + "\n")
        for i in range(len(preds)):
            F.write(str(int(out_label_ids[i])) + ", " + str(int(preds[i])) + "\n")
        F.close()

    preds = None
    out_label_ids = None

    evalset = GDataset(G[len_list[0]+len_list[1]:], sentences[(len_list[0]+len_list[1])*2:],
                       labels[len_list[0]+len_list[1]:])
    evaldata_loader = DataLoader(evalset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    for iter, data_batch in enumerate(evaldata_loader):
        network.eval()
        with torch.no_grad():
            logit, label = network(data_batch)
            # loss = network.loss_fct(logit.view(-1, network.num_labels), label.view(-1))
            # eval_loss += loss.detach().item()
            if preds is None:
                preds = logit.detach().cpu().numpy()
                out_label_ids = label.detach().cpu().numpy()
            else:
                preds = np.append(preds, logit.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, label.detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)

    result = compute_metrics(preds, out_label_ids)

    for key in sorted(result.keys()):
        print("mismatched %s = %s\n" % (key, str(result[key])))

    if result_rec_folder != None:
        F = open(os.path.join(result_rec_folder, ("mismatched_results_print" + text + ".txt")), "w")
        for key in sorted(result.keys()):
            F.write(key + " = " + str(result[key]) + "\n")
        for i in range(len(preds)):
            F.write(str(int(out_label_ids[i])) + ", " + str(int(preds[i])) + "\n")
        F.close()


def main():
    hyps = {}
    hyps["device"] = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    hyps["encoder_type"] = "bert" # "lstm", "bert" or "roberta"
    hyps["use_gnn"] = True # False

    hyps["num_labels"] = 3
    hyps["attn_size"] = 512
    hyps["classifier_hid_dim"] = 512
    hyps["epochs"] = 3
    hyps["lr"] = 1e-4  # 5e-3, 1e-5, 2e-5, 1e-4, 2e-4, 5e-4, 1e-3
    hyps["batch_size"] = 64
    hyps["max_grad_norm"] = 1.0
    hyps["weight_decay"] = 0 # 1e-3
    hyps["adam_epsilon"] = 1e-8
    hyps["train_len"] = 392702
    hyps["warmup_steps"] = int((360000 / hyps["batch_size"]) / 10)
    hyps["dropout_inter"] = 0
    hyps["input_data_list"] = ['data/MNLI/dependency_mnli_train.tsv', 'data/MNLI/dependency_mnli_dev_matched.tsv',
                               'data/MNLI/dependency_mnli_dev_mismatched.tsv']
    hyps["input_label_list"] = ['data/MNLI/train_labels.txt', 'data/MNLI/dev_matched_labels.txt',
                                'data/MNLI/dev_mismatched_labels.txt']
    hyps["save_dependency_pt"] = False
    hyps["rec_result"] = True  # whether to record training and evaluation results
    hyps["mutual_link"] = "co_attn"  # "co_attn", "same_word", "no_link"

    hyps["eval_on_hans"] = False  # whether to evaluate the model on HANS dataset
    hyps["eval_model_folder"] = "results/bert_co_attn/seed69/lr1e-04_202110271729"  # where to load the evaluated model
    

    seed = 69

    hyps["encoder_type"] = hyps["encoder_type"].strip()

    if hyps["encoder_type"] == "bert" or hyps["encoder_type"] == "roberta":
        hyps["embed_size"] = 768
        hyps["encoder_output_size"] = 768
    elif hyps["encoder_type"] == "lstm":
        hyps["embed_size"] = 300
        hyps["encoder_output_size"] = 1024 # 2048

    if hyps["use_gnn"] == True:
        hyps["gnn"] = {}
        hyps["gnn"]["edge_types"] = 4
        hyps["gnn"]["gcn_layers"] = 3  # 3, 5
        hyps["gnn"]["in_features"] = hyps["encoder_output_size"]
        hyps["gnn"]["out_features"] = hyps["encoder_output_size"]
        # hyps["gnn"]["sa_dim"] = embed_size
        hyps["gnn"]["gcn_dp"] = 0
        hyps["gnn"]["gcn_use_bn"] = True
        hyps["gnn"]["use_highway"] = True
       # hyps["gnn"]["out_trans_dim"] = 512

    if hyps["rec_result"]:
        upper_folder = "results/" + hyps["encoder_type"]

        if not hyps["use_gnn"]:
            upper_folder += "_without_gnn"
        else:
            upper_folder += "_" + hyps["mutual_link"]

        upper_folder = upper_folder + "/seed" + str(seed)
        if not os.path.exists(upper_folder):
            os.mkdir(upper_folder)
        if not hyps["eval_on_hans"]:
            result_rec_folder = upper_folder + '/lr' + '{:.0e}'.format(hyps["lr"]) + '_' + time.strftime("%Y%m%d%H%M")
            os.mkdir(result_rec_folder)
        else:
            result_rec_folder = hyps["eval_model_folder"]
    else:
        result_rec_folder = None

    if hyps["eval_on_hans"]:
        if not hyps["rec_result"]:
            print("Please record result first if evaluation on hans is needed.")
            return      
        hyps["eval_model_path"] = os.path.join(result_rec_folder, "model_para2.pkl")
        hyps["eval_hans_result"] = os.path.join(result_rec_folder, "hans_preds_recur.txt")
        hyps["eval_data_list"] = "HANS/dependency_hans.tsv"
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    sentences, adj, labels, len_list = fetch_data_and_graph(hyps)
    # hyps["input_data_list"], hyps["input_label_list"], hyps["edge_types"])

    trainset = GDataset(adj[:hyps["train_len"]], sentences[:hyps["train_len"] * 2], labels[:hyps["train_len"]])
    traindata_loader = DataLoader(trainset, batch_size=hyps["batch_size"], shuffle=False, collate_fn=collate)
    t_total = int(360000 / hyps["batch_size"]) * hyps["epochs"]

    network = MatchNetwork(hyps, t_total)

    if hyps["eval_on_hans"]:
        network.load_state_dict(torch.load(hyps["eval_model_path"]))
        evaluate(network, adj, sentences, labels, len_list, hyps, hyps["eval_model_folder"])
        return

    assert(hyps["train_len"] == len_list[0])
    epoch_losses = []

    print("Training Starts")
    for epoch in tqdm(range(hyps["epochs"]), desc="Training Epochs"):
        network.train()
        epoch_loss = 0
        for iter, data_batch in enumerate(traindata_loader):
            # if iter > 3:
            #     break

            loss = network.loss(data_batch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(network.parameters(), hyps["max_grad_norm"])
            if hyps["encoder_type"] == "bert":
                network.optimizer_bert.step()
            network.optimizer_other.step()
            if hyps["encoder_type"] == "bert":                
                network.scheduler_bert.step()
                network.optimizer_bert.zero_grad()
            network.optimizer_other.zero_grad()

            if iter % 100 == 0:
                # print('Epoch {}, iter {}, loss {:.4f}'.format(epoch, iter, loss.detach().item()))
                print('Epoch {}, iter {}, loss {:.4f}, bert_lr {:.3e}, lr {:.3e}'.format(epoch, iter, loss.detach().item(),
                        network.optimizer_bert.state_dict()['param_groups'][0]['lr'],
                        network.optimizer_other.state_dict()['param_groups'][0]['lr']))

        evaluate(network, adj, sentences, labels, len_list, hyps, result_rec_folder, str(epoch))
        epoch_loss /= (iter + 1)
        # print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)

        if result_rec_folder != None:
            torch.save(network.state_dict(), os.path.join(result_rec_folder, "model_para" + str(epoch) + ".pkl"))
        gc.collect()
        
        network.scheduler_other.step()


import datetime
import math
if __name__ == "__main__":
    start = datetime.datetime.now()
    print("~~~~~~~~ Start Time: " + str(start))
    main()
    end = datetime.datetime.now()
    print("~~~~~~~~ End Time: " + str(end))

    def changeTime(seconds):
        day = 24 * 60 * 60
        hour = 60 * 60
        min = 60
        if seconds < 60:
            return "%d sec" % math.ceil(seconds)
        elif seconds > day:
            days = divmod(seconds, day)
            return "%d day, %s" % (int(days[0]), changeTime(days[1]))
        elif seconds > hour:
            hours = divmod(seconds, hour)
            return "%d hour, %s" % (int(hours[0]), changeTime(hours[1]))
        else:
            mins = divmod(seconds, min)
            return "%d min, %d sec" % (int(mins[0]), math.ceil(mins[1]))

    print("~~~~~~~~ Finally Time Cost: " + changeTime((end-start).seconds))

