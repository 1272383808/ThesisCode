# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Fine-tuning on A Classification Task with pretrained Transformer """

import itertools
import csv
import fire

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


import tokenization
import models
import optim
import train

from utils import set_seeds, get_device, truncate_tokens_pair
from util.get_tokens import *
from util.mapping import *


class CsvDataset(Dataset):
    """ Dataset Class for CSV file """
    labels = None
    def __init__(self, file, pipeline=[]): # cvs file and pipeline object
        Dataset.__init__(self)
        data = []
        with open(file, "r") as f:
            # list of splitted lines : line is also list
            lines = csv.reader(f, delimiter='\t', quotechar=None)
            for instance in self.get_instances(lines): # instance : tuple of fields
                print(instance)
                for proc in pipeline: # a bunch of pre-processing
                    instance = proc(instance)
                data.append(instance)

        # To Tensors
        self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def get_instances(self, lines):
        """ get instance array from (csv-separated) line list """
        raise NotImplementedError


class TxtDataset(Dataset):
    """ Dataset Class for CSV file """
    labels = None
    def __init__(self, file, pipeline=[]): # cvs file and pipeline object
        Dataset.__init__(self)
        data = []
        slicelists = self.dataProcessing(file)

        for instance in self.get_instances(slicelists): # instance : tuple of fields
            for proc in pipeline: # a bunch of pre-processing
                instance = proc(instance)     # (input_ids, segment_ids, input_mask, label_id)
            data.append(instance)

        # To Tensors
        self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

    @classmethod
    def dataProcessing(self, file):
        f = open(file, 'r')
        slicelists = f.read().split("------------------------------")  # 按切片分割
        f.close()

        if slicelists[0] == '':  # 删除切片文件，前后的冗余信息
            del slicelists[0]
        if slicelists[-1] == '' or slicelists[-1] == '\n' or slicelists[-1] == '\r\n':
            del slicelists[-1]

        # slicelists = slicelists[:80]
        data = []
        index = 0
        for slicelist in slicelists:
            if index == 0:
                sentences = slicelist.split('\n')[1:]  # 以行为单位，对切片进行分割(先不要第一行的信息)
            else:
                sentences = slicelist.split('\n')[2:]  # 以行为单位，对切片进行分割(先不要第一行的信息)
            if sentences[0] == '\r' or sentences[0] == '':  # 删除每一个切片前后无关的信息
                del sentences[0]
            if sentences == []:
                continue
            if sentences[-1] == '':
                del sentences[-1]
            if sentences[-1] == '\r':
                del sentences[-1]

            label = str(sentences[-1].strip())
            slice_corpus = []
            for sentence in sentences:  # 对每个切片的每一行进行单独处理
                list_tokens = create_tokens(sentence)
                slice_corpus.append(list_tokens)
            if index % 100 == 0:
                print("slicelist", index, ":", label)
            sentencesAll = ''  # 记录每个切片（名字替换后）的所有token
            slice_corpus, slice_func = mapping(slice_corpus)
            for s in slice_corpus:
                # print(s)
                if sentencesAll == '':
                    sentencesAll = s
                else:
                    sentencesAll = sentencesAll + ' ' + s
            data.append([sentencesAll, label])

            index += 1
        return data

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def get_instances(self, lines):
        """ get instance array from (csv-separated) line list """
        raise NotImplementedError

class CPPCODE(TxtDataset):
    """ Dataset class for SyseVR """
    labels = ("0", "1") # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 0, None): # skip header
            yield line[1], line[0], None # label, text_a, text_b

class MRPC(CsvDataset):
    """ Dataset class for MRPC """
    labels = ("0", "1") # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[3], line[1], line[2] # label, text_a, text_b


class MNLI(CsvDataset):
    """ Dataset class for MNLI """
    labels = ("contradiction", "entailment", "neutral") # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[-1], line[8], line[9] # label, text_a, text_b


def dataset_class(task):
    """ Mapping from task string to Dataset Class """
    table = {'news': MRPC, 'mnli': MNLI, 'cppcode': CPPCODE}
    return table[task]


class Pipeline():
    """ Preprocess Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Tokenizing(Pipeline):
    """ Tokenizing sentence pair """
    def __init__(self, preprocessor, tokenize):
        super().__init__()
        self.preprocessor = preprocessor # e.g. text normalization
        self.tokenize = tokenize # tokenize function

    def __call__(self, instance):
        label, text_a, text_b = instance

        label = self.preprocessor(label)
        tokens_a = self.tokenize(self.preprocessor(text_a))
        tokens_b = self.tokenize(self.preprocessor(text_b)) \
                   if text_b else []

        return (label, tokens_a, tokens_b)


class AddSpecialTokensWithTruncation(Pipeline):
    """ Add special tokens [CLS], [SEP] with truncation """
    def __init__(self, max_len=512):
        super().__init__()
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        # -3 special tokens for [CLS] text_a [SEP] text_b [SEP]
        # -2 special tokens for [CLS] text_a [SEP]
        _max_len = self.max_len - 3 if tokens_b else self.max_len - 2
        truncate_tokens_pair(tokens_a, tokens_b, _max_len)

        # Add Special Tokens
        tokens_a = ['<s>'] + tokens_a + ['</s>']
        tokens_b = tokens_b + ['</s>'] if tokens_b else []
        # tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        # tokens_b = tokens_b + ['[SEP]'] if tokens_b else []

        return (label, tokens_a, tokens_b)


class TokenIndexing(Pipeline):
    """ Convert tokens into token indexes and do zero-padding """
    def __init__(self, indexer, labels, max_len=512):
        super().__init__()
        self.indexer = indexer # function : tokens to indexes
        # map from a label name to a label index
        self.label_map = {name: i for i, name in enumerate(labels)}
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        input_ids = self.indexer(tokens_a + tokens_b)
        segment_ids = [0]*len(tokens_a) + [1]*len(tokens_b) # token type ids
        input_mask = [1]*(len(tokens_a) + len(tokens_b))

        label_id = self.label_map[label]

        # zero padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, label_id)


class Classifier(nn.Module):
    """ Classifier with Transformer """
    def __init__(self, cfg, n_labels):
        super().__init__()
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.transformer(input_ids, segment_ids, input_mask)  # 8 500 768
        # only use the first h in the sequence
        pooled_h = self.activ(self.fc(h[:, 0]))  # 表示在全部数组（维）中取得第0个数据 8 1 768
        logits = self.classifier(self.drop(pooled_h))
        return logits

# total_steps 需要设置，不然最多迭代100000次就停止了
# pretrain_file='./BERT_BASE_DIR/codebert-cpp/ckpt/codebert_cpp_model.ckpt',
# model_file='./output/AU/model_steps_126690.pt',
def main(task='cppcode',
         train_cfg='config/train_mrpc.json',
         model_cfg='config/bert_base.json',
         data_file='./slicetxt/APIfunctionCall/AP_dev.txt',
         model_file='./output/FC/model_steps_36000.pt',
         pretrain_file='./BERT_BASE_DIR/codebert-cpp/ckpt/codebert_cpp_model.ckpt',
         data_parallel=True,
         vocab='./BERT_BASE_DIR/codebert-cpp/vocab.json',
         save_dir='./output/FC',
         max_len=500,
         mode='eval'):

# def main(task='news',
#          train_cfg='config/train_mrpc.json',
#          model_cfg='config/bert_base.json',
#          data_file='./slicetxt/WNLI/train.tsv',
#          model_file=None,
#          pretrain_file='./BERT_BASE_DIR/uncased_L-12_H-768_A-12/bert_model.ckpt',
#          data_parallel=True,
#          vocab='./BERT_BASE_DIR/uncased_L-12_H-768_A-12/vocab.txt',
#          save_dir='./output/news',
#          max_len=128,
#          mode='eval'):

    cfg = train.Config.from_json(train_cfg)
    model_cfg = models.Config.from_json(model_cfg)

    set_seeds(cfg.seed)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    TaskDataset = dataset_class(task)  # task dataset class according to the task
    pipeline = [Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
                AddSpecialTokensWithTruncation(max_len),
                TokenIndexing(tokenizer.convert_tokens_to_ids,
                              TaskDataset.labels, max_len)]
    dataset = TaskDataset(data_file, pipeline)
    data_iter = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = Classifier(model_cfg, len(TaskDataset.labels))
    # print(model)
    criterion = nn.CrossEntropyLoss()

    trainer = train.Trainer(cfg,
                            model,
                            data_iter,
                            optim.optim4GPU(cfg, model),
                            save_dir, get_device())

    if mode == 'train':
        def get_loss(model, batch, global_step): # make sure loss is a scalar tensor
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            _, label_pred = logits.max(1)
            result = (label_pred == label_id).float()
            accuracy = result.mean()
            loss = criterion(logits, label_id)
            return loss, accuracy

        trainer.train(get_loss, model_file, pretrain_file, data_parallel)

    elif mode == 'eval':
        def evaluate(model, batch):    # 对于一个batch数据的处理
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            _, label_pred = logits.max(1)
            result = (label_pred == label_id).float() #.cpu().numpy()
            accuracy = result.mean()
            # print("predict:", label_pred)
            # print("label_id:", label_id)
            return accuracy, result, label_pred, label_id

        results = trainer.eval(evaluate, model_file, data_parallel)
        total_accuracy = torch.cat(results).mean().item()
        print('Accuracy:', total_accuracy)


if __name__ == '__main__':
    fire.Fire(main)
