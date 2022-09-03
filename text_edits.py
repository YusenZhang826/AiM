import sys, os
import json

# from transformers import BertTokenizer

label2act = {0: 'O', 1: 'B-sub', 2: 'B-del', 3: 'B-add', 4: 'I-sub', 5: 'I-del', 6: 'pad'}
act2label = {"O": 0, "B-sub": 1, "B-del": 2, "B-add": 3, "I-sub": 4, "I-del": 5, "pad":6}
for k, v in label2act.items():
    act2label[v] = k


def calc_edit_distance(lta, ltb):
    # base -> lta
    N = len(lta)
    M = len(ltb)

    dp = []
    prv = []
    for i in range(N):
        dp.append([100000] * M)
        prv.append([None] * M)

    # Init
    if lta[0] == ltb[0]:
        dp[0][0] = 0
        prv[0][0] = 'yes'
    else:
        dp[0][0] = 1
        prv[0][0] = 'replace'

    for i in range(N):
        for j in range(M):
            if i == 0 and j == 0:
                continue
            # print(dp[i][j])
            if j - 1 >= 0 and dp[i][j] > dp[i][j - 1] + 1:
                dp[i][j] = dp[i][j - 1] + 1
                prv[i][j] = 'add'
            if i - 1 >= 0 and dp[i][j] > dp[i - 1][j] + 1:
                dp[i][j] = dp[i - 1][j] + 1
                prv[i][j] = 'delete'
            if i - 1 >= 0 and j - 1 >= 0 and dp[i][j] > dp[i - 1][j - 1] + 1:
                dp[i][j] = dp[i - 1][j - 1] + 1
                prv[i][j] = 'replace'
            if lta[i] == ltb[j]:
                prv[i][j] = 'yes'
                if i > 0 and j > 0:  # and dp[i][j] >= dp[i-1][j-1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = max(i, j)
            # print(i, j, lta[i] == ltb[j], dp[i][j], prv[i][j])

    # print('-'* 50)

    dist = dp[N - 1][M - 1]
    edits = []
    x, y = N - 1, M - 1

    while x >= 0 and y >= 0 and prv[x][y] is not None:
        # print(x, y, dp[x][y], prv[x][y])
        if prv[x][y] == 'yes':
            x -= 1
            y -= 1
        elif prv[x][y] == 'replace':
            edits.append((x, y, lta[x], ltb[y]))
            x -= 1
            y -= 1
        elif prv[x][y] == 'add':
            edits.append((x, y, '', ltb[y]))
            y -= 1
        elif prv[x][y] == 'delete':
            edits.append((x, y, lta[x], ''))
            x -= 1
        else:
            break
    while x >= 0:
        edits.append((x, -1, lta[x], ''))
        x -= 1
    while y >= 0:
        edits.append((-1, y, '', ltb[y]))
        y -= 1

    edits = list(reversed(edits))
    merged_edits = []
    for x, y, cx, cy in edits:
        if len(merged_edits) == 0:
            merged_edits.append([x, y, cx, cy])
            continue
        if cy == '' and y == merged_edits[-1][1]:
            merged_edits[-1][2] += cx
        elif cx == '' and x == merged_edits[-1][0]:
            merged_edits[-1][3] += cy
        else:
            merged_edits.append([x, y, cx, cy])

    return dist, edits, merged_edits


def generate_edit_labels(text: str, img_grd: str, act2label: dict = act2label):
    a = calc_edit_distance(text, img_grd)
    merged = a[-1]
    pad_blk = False
    labels = [act2label['O']] * len(text)
    for j in merged:
        idx = j[0]
        if idx == -1:
            pad_blk = True
            continue
        char_change = j[2]
        change_res = j[3]
        if len(char_change) == len(change_res):
            sub_lens = len(char_change)
            labels[idx] = act2label['B-sub']
            for i in range(idx+1, idx+sub_lens):
                labels[i] = act2label['I-sub']
        elif len(char_change) < len(change_res):
            if char_change == '':
                labels[idx] = act2label['B-add']
            else:
                labels[idx] = act2label['B-sub']
                lens = len(char_change)
                for i in range(idx + 1, idx + lens):
                    labels[i] = act2label['I-sub']

        elif len(char_change) > len(change_res):
            if change_res == '':
                lens = len(char_change)
                labels[idx] = act2label['B-del']
                for i in range(idx+1, idx + lens):
                    labels[i] = act2label['I-del']
            else:
                labels[idx] = act2label['B-sub']
                lens = len(char_change)
                for i in range(idx + 1, idx + lens):
                    labels[i] = act2label['I-sub']
    text = '<BLK>' + text
    if pad_blk:
        labels = [act2label['B-add']] + labels
    else:
        labels = [act2label['O']] + labels
    labels = merge_consecutive_label(labels)
    return labels, text


def generate_bin_labels(text, img_grd):
    labels = [0] * len(text)
    for i, t in enumerate(text):
        if t in img_grd:
            labels[i] = 1
        else:
            labels[i] = 0

    return labels

def merge_consecutive_label(edit_labels:list):
    out = edit_labels.copy()
    lst = []
    i = 0
    b_idx = e_idx = -1
    while i < len(edit_labels):
        if edit_labels[i] == act2label['B-sub']:
            b_idx = i
            j = i
            while j < len(edit_labels) and edit_labels[j] == act2label['B-sub']:
                j += 1

            i = j + 1
            e_idx = j if j != i else -1

        if b_idx != -1 and e_idx != -1:
            lst.append((b_idx, e_idx))
            b_idx = e_idx = -1
        i += 1

    for b, e in lst:
        for idx in range(b+1, e):
            out[idx] = act2label['I-sub']
    return out

def generate_edit_data(data_file):
    data_json = load_json(data_file)
    edit_label_data = []
    for data in data_json:
        new_data = data.copy()
        img_grd = new_data['handwriting']
        text = new_data['answer']
        text = clear_punc(text)
        if '\\b' in img_grd or '/b' in img_grd:
            continue
        img_grd = clear_punc(img_grd)
        img_grd = img_grd.replace('\x00', '')

        edit_labels, merged = generate_edit_labels(text, img_grd, act2label)
        bin_labels = generate_bin_labels(text, img_grd)
        assert len(edit_labels) == len(bin_labels) == len(text)
        new_data['answer'] = text
        new_data['handwriting'] = img_grd
        new_data['text_edit_labels'] = edit_labels
        new_data['text_bin_labels'] = bin_labels
        new_data['merged result'] = merged
        edit_label_data.append(new_data)

    main_path = './eink_data_edit_label/'
    save_path = data_file.split('/')[-1].replace('.json', '_edit_label.json')
    save_path = main_path + save_path
    print(save_path)
    save2json(edit_label_data, save_path)
    return edit_label_data


def clear_punc(text):
    punc_token = [',', '.', '\'', '?', '!', ';', ':', '\\', '，', '。', '、', '；', '：', '？', '！',
                  '"', '“', '”', '\u0000', ' ', '\t', '>>', '<<', ' ', '．']
    if '\b' in text or '\\b' in text:
        return text

    for punc in punc_token:
        if punc in text:
            text = text.replace(punc, '')
    return text


def save2json(json_data, save_path):
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, indent=2, ensure_ascii=False)


def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


if __name__ == '__main__':
    bert_path = 'E:/bert-base-chinese'
    # tokenizer = BertTokenizer.from_pretrained(bert_path)
    eink_test = './eink_data/chinese_data.eink.test.json'
    eink_train = './eink_data/chinese_data.eink.train.json'
    eink_dev = './eink_data/dev_set.json'
    eink_neg = 'E:/neg_data/neg_data_info_wo_dev.json'
    eink_pos = 'E:/pos_data/pos_data_info_wo_dev.json'
    hwdb_2x_train = './hwdb_data/HWDB2.x_train_data.json'
    hwdb_2x_test = './hwdb_data/HWDB2.x_test_data.json'
    eink_img_absolute_path = '/data/cfs_ai_judger/neutrali/data_factory/filling_data/'

    files2generate = [eink_train, eink_test, eink_pos, eink_neg, eink_dev]
    for f in files2generate:
        generate_edit_data(f)
