import cv2
from math import ceil
from matplotlib import pyplot as plt
import collections
from tqdm import tqdm
import random
from text_edits import generate_bin_labels, generate_edit_labels, calc_edit_distance, act2label, label2act
from dataset_seqlabel_for_edit import get_char_list, save2json, load_json
from text_edits import clear_punc


chars_list, _ = get_char_list()

max_len = 52

similar_shape_dict = load_json('./eink_data/SimilarShape.txt')
print('size of similar shape dict', len(similar_shape_dict))


def get_frequent_used():
    file = './eink_data/frequently_used.txt'
    freq_list = ''
    with open(file, 'r', encoding='utf-8') as f:
        for ch in f.readlines():
            ch = ch.replace('\n', '').replace('\t', '')
            freq_list += ch

    return freq_list


freq_list = get_frequent_used()


def merge_consecutive_label(edit_labels: list):
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
        for idx in range(b + 1, e):
            out[idx] = act2label['I-sub']
    return out


def get_random_chars(chars_list: str, handwrite: str, in_hand=False):
    rand_char = random.choice(list(chars_list))
    if not in_hand:
        while rand_char in handwrite:
            rand_char = random.choice(list(chars_list))
    else:
        while rand_char not in handwrite:
            rand_char = random.choice(list(handwrite))

    return rand_char


def fulfill_source_data(source_data: dict):
    hand_write = source_data['handwriting']
    if 'answer' in source_data:
        answer = source_data['answer']
        source_data['answer'] = clear_punc(answer)
        source_data['handwriting'] = clear_punc(hand_write)
        # edit_labels = source_data['text_edit_labels']
        # bin_labels = source_data['text_bin_labels']
        # edit_mean_ = [label2act[c] for c in edit_labels]
        # source_data['edit_means'] = edit_mean_
    else:
        hand_write = clear_punc(hand_write)
        answer = hand_write
        source_data['handwriting'] = hand_write
        bin_labels = [1] * len(answer)
        edit_labels = [0] * len(answer)
        source_data['answer'] = answer
        source_data['text_bin_labels'] = bin_labels
        source_data['text_edit_labels'] = edit_labels
        edit_mean_ = [label2act[c] for c in edit_labels]
        source_data['edit_means'] = edit_mean_
        source_data['label'] = 1

    return source_data


def get_random_equal_data(source_data: dict, mode='random'):
    rand_data = source_data.copy()
    hand_write = rand_data['handwriting']
    r_answer = rand_data['answer']
    # rand_answer = rand_change_text_to_balance_01(r_answer, r_bin_labels, hand_write)
    rand_answer = rand_change_text(r_answer, hand_write)
    _bin_labels = generate_bin_labels(rand_answer, hand_write)
    _edit_labels, new_answer = generate_edit_labels(rand_answer, hand_write, act2label)
    edit_mean = [label2act[c] for c in _edit_labels]
    rand_data['text_edit_labels'] = _edit_labels
    rand_data['answer'] = new_answer
    rand_data['edit_means'] = edit_mean
    rand_data['label'] = 2

    return rand_data


def get_random_shorter_data(source_data: dict, mode='random'):
    rand_data = source_data.copy()
    hand_write = rand_data['handwriting']
    source_len = len(hand_write)
    shorter = random.randint(1, source_len - 1)
    shorter = min(int((0.5 * source_len)), shorter)
    new_len = source_len - shorter
    hand_write = list(hand_write)
    if mode == 'random':
        while len(hand_write) > new_len:
            idx = random.randint(1, len(hand_write)) - 1
            hand_write.pop(idx)

    rand_answer = ''.join(hand_write)
    rand_hand_write = rand_data['handwriting']
    _edit_labels, new_answer = generate_edit_labels(rand_answer, rand_hand_write)
    edit_mean = [label2act[c] for c in _edit_labels]
    rand_data['text_edit_labels'] = _edit_labels
    rand_data['answer'] = new_answer
    rand_data['edit_means'] = edit_mean
    rand_data['label'] = 2

    return rand_data


def get_random_longer_data(source_data: dict, mode='random'):
    rand_data = source_data.copy()
    hand_write = rand_data['handwriting']
    source_len = len(hand_write)
    if mode == 'random':
        up_longer = ceil(0.6 * source_len)
        longer = random.randint(1, max_len - source_len - 1)
        longer = min(longer, up_longer)
        new_len = source_len + longer
        hand_write = list(hand_write)
        while len(hand_write) < new_len:
            idx = random.randint(1, len(hand_write)) - 1
            char = random.choice(list(freq_list))
            hand_write.insert(idx, char)
    else:
        up_longer = ceil(0.5 * source_len)
        longer = random.randint(1, max_len - source_len - 1)
        longer = min(longer, up_longer)
        new_len = source_len + longer
        hand_write = list(hand_write)
        while len(hand_write) < new_len:
            char = random.choice(list(freq_list))
            hand_write.append(char)

    rand_answer = ''.join(hand_write)
    rand_hand_write = rand_data['handwriting']
    _edit_labels, new_answer = generate_edit_labels(rand_answer, rand_hand_write)
    edit_mean = [label2act[c] for c in _edit_labels]
    rand_data['text_edit_labels'] = _edit_labels
    rand_data['answer'] = new_answer
    rand_data['edit_means'] = edit_mean
    rand_data['label'] = 2

    return rand_data


def rand_change_text(text: str, handwrite: str) -> str:
    ans_len = len(text)
    if ans_len == 1:
        text = get_random_chars(freq_list, handwrite, in_hand=False)
        return text

    text = list(text)
    change_num = random.choice([1, 2, 1, len(text)])
    if change_num != 2:
        idxs = list(range(ans_len))
        idx2cvt = []
        for i in range(change_num):
            idx = random.choice(idxs)
            while idx in idx2cvt:
                idx = random.choice(idxs)
            idx2cvt.append(idx)

        for idx in idx2cvt:
            source_char = text[idx]
            if source_char in similar_shape_dict:
                all_sims = similar_shape_dict[source_char]
                candidates = [s for s in all_sims if s in chars_list]
                if len(candidates) > 0:
                    rand_char = random.choice(candidates)
                else:
                    rand_char = get_random_chars(freq_list, handwrite, in_hand=False)
            else:
                rand_char = get_random_chars(freq_list, handwrite, in_hand=False)
            text[idx] = rand_char

    else:
        i, j = 0, len(text) - 1
        while i < j:
            text[i], text[j] = text[j], text[i]
            i += 1
            j -= 1
        half = len(text) // 2
        text[:half] = text[:half][::-1]
        text[half:] = text[half:][::-1]

    return ''.join(text)


def rand_change_text_to_balance_01(text, bin_labels, handwrite):
    seq_len = len(text)
    text = list(text)
    one_idx = []
    zero_idx = []
    for i, b in enumerate(bin_labels):
        if b == 1:
            one_idx.append(i)
        else:
            zero_idx.append(i)
    num_ones = len(one_idx)
    num_zero = seq_len - num_ones
    bias = abs(num_ones - num_zero)
    num2change = bias // 2
    idx2change = []
    if num_ones > num_zero:  # 换为未出现的字
        for _ in range(num2change):
            ch = random.choice(one_idx)
            while ch in idx2change:
                ch = random.choice(one_idx)
            idx2change.append(ch)

        for idx in idx2change:
            source_char = text[idx]
            if source_char in similar_shape_dict:
                all_sims = similar_shape_dict[source_char]
                candidates = [s for s in all_sims if s in chars_list]
                if len(candidates) > 0:
                    rand_char = random.choice(candidates)
                else:
                    rand_char = get_random_chars(freq_list, handwrite, in_hand=False)
            else:
                rand_char = get_random_chars(freq_list, handwrite, in_hand=False)
            text[idx] = rand_char
    else:
        # 换为已出现字
        for _ in range(num2change):
            ch = random.choice(zero_idx)
            while ch in idx2change:
                ch = random.choice(zero_idx)
            idx2change.append(ch)

        for idx in idx2change:
            rand_char = get_random_chars(freq_list, handwrite, in_hand=True)
            text[idx] = rand_char

    return ''.join(text)


def random_extend_data(file, source='eink', keep=False):
    json_data = load_json(file)
    print(len(json_data))
    assert source in ['eink', 'hwdb']
    save_path = './random_extend_data_for_edit_label_v3/'
    rand_extend_data = []
    for data in tqdm(json_data):
        try:
            img_file = data['file']
            handwrite = data['handwriting']
            # if '\\b' in handwrite or '\b' in handwrite:
            #     continue
            # img_path = 'F:/' + img_file
            # image = cv2.imread(img_path)
            # if len(image.shape) == 3:
            #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # ratio = float(image.shape[1]) / float(image.shape[0])
            #
            # th = height = 128
            # tw = int(th * ratio)
            # if tw > 3783:
            #     continue

            source_data = fulfill_source_data(data)
            source_hand_write = source_data['handwriting']
            if not keep:
                for i in range(random.randint(0, 1)):
                    equal_rand_data = get_random_equal_data(source_data)
                    rand_extend_data.append(equal_rand_data)

                if len(source_hand_write) == 1:
                    longer_rand_data = get_random_longer_data(source_data, mode=source)
                    rand_extend_data.append(longer_rand_data)

                elif len(source_hand_write) == max_len:
                    shorter_rand_data = get_random_shorter_data(source_data, )
                    rand_extend_data.append(shorter_rand_data)
                else:
                    # longer
                    for _ in range(random.randint(0, 1)):
                        longer_rand_data = get_random_longer_data(source_data, mode=source)
                        rand_extend_data.append(longer_rand_data)

                    # shorter
                    for _ in range(random.randint(0, 1)):
                        shorter_rand_data = get_random_shorter_data(source_data)
                        rand_extend_data.append(shorter_rand_data)

            source_edit_label, new_answer = generate_edit_labels(source_data['answer'], source_data['handwriting'])
            # source_edit_label = merge_consecutive_label(source_edit_label)
            source_data['text_edit_labels'] = source_edit_label
            source_data['answer'] = new_answer
            edit_mean = [label2act[c] for c in source_edit_label]
            source_data['edit_means'] = edit_mean
            rand_extend_data.append(source_data)

        except Exception as e:
            print(repr(e))
            continue

    print(len(rand_extend_data))
    check_data_distribution(rand_extend_data)
    file_name = file.split('/')[-1]

    file_name_new = file_name.replace('.json', '_edit_extend.json')
    new_path = save_path + file_name_new
    save2json(rand_extend_data, new_path)

    return rand_extend_data


def check_data_distribution(data_json):
    if isinstance(data_json, str):
        data_json = load_json(data_json)

    label_cnt = collections.defaultdict(int)
    print(len(data_json))
    for d in data_json:
        edit_label = d['text_edit_labels']
        for e in edit_label:
            label_cnt[label2act[e]] += 1
    total = sum(list(label_cnt.values()))
    print([(k, v, v / total) for k, v in label_cnt.items()])
    return label_cnt


def check_img(file: str):
    img = cv2.imread(file)
    h, w, c = img.shape
    for i in range(h):
        for j in range(w):
            for k in range(c):
                pixel = img[i, j, k]
                if pixel != 255 and pixel != 0:
                    print(file)
                    print(pixel)


def get_avg_answer_lens(file):
    if isinstance(file, str):
        data_json = load_json(file)
    else:
        data_json = file
    mode = '_train' if len(data_json) > 2000 else '_test'
    avg = 0
    max_l = 0
    min_l = 100
    cnt = collections.defaultdict(int)
    lc = collections.defaultdict(int)
    for d in data_json:
        answer = d['handwriting']
        # answer = clear_punc(answer)
        label = d['label']
        lc[label] += 1
        l = len(answer)
        cnt[l] += 1
        max_l = max(max_l, l)
        min_l = min(min_l, l)
        avg += l
    avg /= len(data_json)
    items = sorted(cnt.items(), key=lambda x: x[0])
    print(lc)

    x = [i[0] for i in items]
    y = [i[1] for i in items]
    plt.figure()
    plt.xlabel('hands lens')
    plt.ylabel('num')
    plt.title('distribute_extend' + mode)
    plt.plot(x, y)

    plt.show()

    return avg, max_l, min_l





if __name__ == '__main__':

    eink_new_train = './eink_data/chinese_data.eink.train.new_v2.json'
    eink_new_test = './eink_data/chinese_data.eink.test.new_v2.json'
    t = random_extend_data(eink_new_test, keep=True)


