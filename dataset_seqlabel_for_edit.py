from torchvision import transforms
import argparse
import json
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from transformers import BertTokenizer

from transformers import BertTokenizer
bert_path = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_path)

parser = argparse.ArgumentParser()
train_file = 'jsonfile/chinese_data.eink.train.new_v2_edit_extend.json'
# test_file = '/data/cfs_ai_judger/laxzhang/ft_local/ft_local/random_extend_data_for_edit_label/chinese_data.eink.test.new_v2_edit_extend.json'
test_file = 'jsonfile/chinese_data.eink.test.new_v2_edit_extend.json'
# test_file = '/data/cfs_ai_judger/laxzhang/ft_local/ft_local/random_extend_data_for_edit_label/chinese_data.eink.test_edit_label_extend.json'
# test_file = '/data/cfs_ai_judger/laxzhang/ft_local/ft_local/random_extend_data_for_edit_label/eink_whole_data_edit.json'
# test_file = '/data/cfs_ai_judger/laxzhang/ft_local/ft_local/random_extend_data_for_edit_label/chinese_eink.test.resplit_extend.json'
# eink_img_absolute_path = '/data/cfs_ai_judger/neutrali/data_factory/filling_data/'
eink_img_absolute_path = '../../paddle_ocr_test/'

eink_neg_file = '/data/cfs_ai_judger/laxzhang/ft_local/ft_local/random_extend_data_for_edit_label/neg_data_info_wo_dev_edit_label_extend.json'
eink_neg_path = '/data/cfs_ai_judger/laxzhang/ft_local/neg_data/'
eink_pos_file = '/data/cfs_ai_judger/laxzhang/ft_local/ft_local/random_extend_data_for_edit_label/pos_data_info_wo_dev_edit_label_extend.json'
eink_pos_path = '/data/cfs_ai_judger/laxzhang/ft_local/pos_data/'
# eink_neg_path = '/data/cfs_ai_judger/laxzhang/ft_local/neg_data/'
# eink_neg_file = '/data/cfs_ai_judger/laxzhang/ft_local/ft_local/neg_data_info_wo_dev.json'
# eink_pos_path = '/data/cfs_ai_judger/laxzhang/ft_local/pos_data/'
# eink_pos_file = '/data/cfs_ai_judger/laxzhang/ft_local/ft_local/pos_data_info_wo_dev.json'
hwdb_2_x_train_img_path = '/data/cfs_ai_judger/laxzhang/ft_local/ft_local/HWDB2.xTrain_images/'
hwdb_2_x_train_json = '/data/cfs_ai_judger/laxzhang/ft_local/ft_local/random_extend_data_for_edit_label/HWDB2.x_train_data_extend.json'
hwdb_2_x_test_img_path = '/data/cfs_ai_judger/laxzhang/ft_local/ft_local/HWDB2.xTest_images/'
hwdb_2_x_test_json = '/data/cfs_ai_judger/laxzhang/ft_local/ft_local/random_extend_data_for_edit_label/HWDB2.x_test_data_extend.json'

ch_crop_json = '/data/cfs_ai_judger/laxzhang/ft_local/ft_local/random_extend_data_for_edit_label/chinese_content_extend.json'
ch_crop_path = '/data/cfs_ai_judger/laxzhang/ft_local/ft_local/ch_crop/'

ch_crop_test_json = '/data/cfs_ai_judger/laxzhang/ft_local/ft_local/random_extend_data_for_edit_label/ch_crop_test_resource.json'
ch_crop_train_json = '/data/cfs_ai_judger/laxzhang/ft_local/ft_local/random_extend_data_for_edit_label/ch_crop_train_resource.json'

dev_file = '/data/cfs_ai_judger/laxzhang/ft_local/ft_local/random_extend_data_for_edit_label/dev_set_edit_label_extend.json'
dev_ratio = 0.2

img_base_path = [eink_img_absolute_path, eink_img_absolute_path, eink_neg_path, eink_pos_path, '']
json_path = [train_file, test_file, eink_neg_file, eink_pos_file, dev_file]

label2act = {0: 'O', 1: 'B-sub', 2: 'B-del', 3: 'B-add', 4: 'I-sub', 5: 'I-del', 6: 'pad'}
act2label = {}
for k, v in label2act.items():
    act2label[v] = k
tokenizer = BertTokenizer.from_pretrained(bert_path)


class seqlables_dataset_for_edit():
    def __init__(self, json_files, img_base_path, max_width, max_length, chars_list,
                 tokenizer: BertTokenizer, label_type='text_edit_labels', fast=False, mode='train'):
        self.normalize = transforms.Normalize((0.5), (0.51))
        self.imgs = []
        self.hand_write = []
        self.answer = []
        self.text_ids = []
        self.text_lens = []
        self.labels = []
        self.text_att_masks = []
        self.img_att_masks = []
        self.label_mask = []
        self.judge_labels = []
        self.chars_list = chars_list
        self.max_width = max_width
        self.max_length = max_length
        self.label_type = label_type
        self.tokenizer = tokenizer
        assert label_type in ['text_bin_labels', 'text_edit_labels']
        self.label_pad = 0 if label_type == 'text_bin_labels' else act2label['pad']
        self.get_dict()

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])

        for json_file, img_base in zip(json_files, img_base_path):
            with open(json_file, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)

            thre = 1000000

            for d in data_dict:
                if fast and mode == 'train':
                    thre = 1e6
                    if len(self.imgs) > thre:
                        break
                if fast and mode == 'test':
                    thre = 1000
                    if len(self.imgs) > thre:
                        break
                if fast and mode == 'dev':
                    thre = 1e6
                    if len(self.imgs) > thre:
                        break
                if fast and mode == 'hwdb':
                    thre = 1e6
                    if len(self.imgs) > thre:
                        break

                img_path = d['file']
                if mode == 'dev':
                    pass
                else:
                    img_path = img_base + img_path

                hand_writing = d['handwriting']
                hand_writing = hand_writing.replace('\x00', '')
                if '\\b' in hand_writing or '/b' in hand_writing:
                    continue
                answer = d['answer']
                judge_label = d['label'] if 'label' in d else -100
                self.judge_labels.append(judge_label)
                self.hand_write.append(hand_writing)
                self.answer.append(answer)
                answer = list(answer)
                text_id = self.encode(answer)

                text_len = len(text_id)
                text_att_mask = [1] * len(text_id)
                seq_label_mask = text_att_mask
                seq_labels = d[label_type]

                text_att_mask = text_att_mask
                seq_label_mask = seq_label_mask

                # answer = self.padding(answer, max_len=self.max_length, pad_idx='[PAD]')

                # print(hand_writing)
                #                 print(answer, len(answer))
                #                 print(text_id, len(text_id))
                #                 print(text_att_mask, len(text_att_mask))
                #                 print(seq_label_mask, len(seq_label_mask))
                #                 print(seq_labels, len(seq_labels))

                text_id = self.padding(text_id, self.max_length, 0)

                self.imgs.append(img_path)
                self.text_ids.append(text_id)
                self.text_lens.append(text_len)
                self.labels.append(seq_labels)
                self.text_att_masks.append(text_att_mask)
                self.label_mask.append(seq_label_mask)

                # self.hand_write.append(hand_writing)

    def __len__(self):
        assert len(self.imgs) == len(self.answer) == len(self.text_ids) == len(self.labels) == len(self.text_att_masks)
        return len(self.imgs)

    def get_dict(self):
        self.dict = {}
        self.dict['<BLK>'] = 0
        for i, char in enumerate(self.chars_list):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1

        self.dict['<UNK>'] = len(self.chars_list) - 1

    def encode(self, text):
        text = text[5:]  # remove <BLK>
        token_ids = [0]  # 0 is for <BLK>
        for t in text:
            if t in self.dict:
                token_ids.append(self.dict[t])
            else:
                token_ids.append(self.dict['<UNK>'])

        return token_ids

    def padding(self, seq, max_len, pad_idx):
        if len(seq) >= max_len:
            return seq
        seq = seq + [pad_idx] * (max_len - len(seq))
        return seq

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        # print('*******************', img_path)
        img_mask = [0] * (self.max_width // 32)  # length of img features
        image = cv2.imread(img_path)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #             _, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
        ratio = float(image.shape[1]) / float(image.shape[0])

        th = height = 128
        tw = int(th * ratio)

        rsz = cv2.resize(image, (tw, th), fx=0, fy=0, interpolation=cv2.INTER_AREA)
        img_data, width = rsz, tw
        truely_img_len = width // 32  # generate img mask
        if width % 32 != 0:
            truely_img_len += 1
        for i in range(truely_img_len):
            img_mask[i] = 1
        pad_data = np.zeros((height, self.max_width - width), dtype=np.uint8)
        pad_data.fill(255)
        rsz = np.concatenate((img_data, pad_data), axis=1)
        width = rsz.shape[1]
        rsz = self.transforms(rsz)

        text_ids = self.text_ids[idx]
        seq_label = self.labels[idx]
        text_att_mask = self.text_att_masks[idx]
        seq_label_mask = self.label_mask[idx]

        text_ids = self.padding(text_ids, max_len=self.max_length, pad_idx=0)
        text_att_mask = self.padding(text_att_mask, max_len=self.max_length, pad_idx=0)
        seq_label = self.padding(seq_label, max_len=self.max_length, pad_idx=self.label_pad)
        seq_label_mask = self.padding(seq_label_mask, max_len=self.max_length, pad_idx=0)

        assert len(text_ids) == len(text_att_mask) == len(seq_label) == len(seq_label_mask)

        return rsz, np.array(img_mask), np.array(text_ids), np.array(text_att_mask), np.array(seq_label_mask), np.array(
            seq_label), self.hand_write[idx], self.answer[idx], np.array(self.judge_labels[idx]), self.imgs[idx]


def get_char_list():
    chars_path = 'final_chars_list.txt'
    chars_list = ''
    with open(chars_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            char = line[0]
            if char not in chars_list:
                chars_list = chars_list + char

    num_words = len(chars_list)
    return chars_list, num_words


def get_gloable_max_width(base_path, json_path):
    all_imgs = []
    for base, json_file in zip(base_path, json_path):
        # print(base, json_file)
        with open(json_file, 'r', encoding='utf-8') as f:
            train_dict = json.load(f)
            for d in train_dict:
                img_path = d['file']
                img_path = base + img_path
                all_imgs.append(img_path)

    max_width = 0
    print('num imgs', len(all_imgs))
    for i in tqdm(range(len(all_imgs))):
        img_path = all_imgs[i]
        # print(img_path)
        if os.path.exists(img_path):
            image = cv2.imread(img_path)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #                 _, image = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
            ratio = float(image.shape[1]) / float(image.shape[0])
            th = height = 128
            tw = int(th * ratio)
            max_width = max(tw, max_width)
        else:
            continue
    print('max width of imgs', max_width)
    return max_width


def get_max_seq_len(json_files: list):
    max_seq_len = 0
    # all_charaters = []
    # char_list = ''
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
            for d in data_dict:
                hand_write = d['answer']
                max_seq_len = max(max_seq_len, len(hand_write))

    return max_seq_len


def save2json(json_data, save_path):
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, indent=4, ensure_ascii=False)


def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def create_seqlabel_dataset_wo_bert(fast, chars_list, train_list=None):
    if train_list is None:
        train_list = [[train_file], [eink_img_absolute_path]]
    train_json_file, train_img_path = train_list[0], train_list[1]
    # max_width = 2258
    max_width = 3783
    if max_width % 32 != 0:
        max_width = max_width + 32 - (max_width % 32)
    max_length = 52
    # print(max_width)
    train_data = seqlables_dataset_for_edit(train_json_file,
                                            train_img_path,
                                            max_width, max_length, chars_list, tokenizer, fast=fast, mode='train')

    eink_test_data = seqlables_dataset_for_edit([test_file],
                                                [eink_img_absolute_path],
                                                max_width, max_length, chars_list, tokenizer, fast=fast, mode='test')

    # eink_dev_data = seqlables_dataset_for_edit([dev_file], [''], max_width, max_length,
    #                                            chars_list, tokenizer, fast=fast, mode='dev')
    #
    # hwdb_test_data = seqlables_dataset_for_edit([hwdb_2_x_test_json],
    #                                             [hwdb_2_x_test_img_path],
    #                                             max_width, max_length, chars_list, tokenizer, fast=fast, mode='hwdb')


    print('num dataset:', len(train_data), len(eink_test_data))
    # return train_data, eink_test_data, eink_dev_data, hwdb_test_data
    return train_data, eink_test_data

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders


if __name__ == '__main__':
    chars_list, num_words = get_char_list()
    print(num_words)
    datasets = create_seqlabel_dataset_wo_bert(fast=True, chars_list=chars_list)
    for d in datasets:
        print(len(d))

