import os
import pickle
import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from config import Config
from raw_excel_process import get_all_excel_data, get_train_input_label, get_train_input_label2

random.seed(822)
np.random.seed(822)

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        print("tokenizer has been loaded.")
        tokenizer = pickle.load(f)
    return tokenizer

def char_tokenizer(sentence_list, tokenizer_path=None, tokenizer=None, maxLen=Config.sequenceMaxLength):
    if tokenizer:
        idx_sequence_list = tokenizer.texts_to_sequences(sentence_list)
        idx_sequence_pad_list = pad_sequences(idx_sequence_list, maxLen,
                                              padding='post', truncating='post')
        return idx_sequence_pad_list
    if not tokenizer_path:
        print("tokenizer_path and tokenizer can not be None.")
        return []
    if os.path.exists(tokenizer_path):
        # 使用现有的tokenizer
        tokenizer = load_tokenizer(tokenizer_path)
        # print(tokenizer.word_index)
        idx_sequence_list = tokenizer.texts_to_sequences(sentence_list)
        idx_sequence_pad_list = pad_sequences(idx_sequence_list, maxLen,
                                              padding='post', truncating='post')
    else:
        # 建立一个max_features个词的字典
        tokenizer = Tokenizer(num_words=Config.numChars,
                              lower=False,
                              char_level=True)
        # 使用一系列文档来生成token词典，参数为list类，每个元素为一个文档。可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小。
        tokenizer.fit_on_texts(sentence_list)
        #print(tokenizer.word_index)
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("tokenizer has been saved.")
        # 对每个词编码之后，每个文本中的每个词就可以用对应的编码表示，即每条文本已经转变成一个向量了 将多个文档转换为word下标的向量形式,shape为[len(texts)，len(text)] -- (文档数，每条文档的长度)
        idx_sequence_list = tokenizer.texts_to_sequences(sentence_list)
        idx_sequence_pad_list = pad_sequences(idx_sequence_list, maxLen,
                                              padding='post', truncating='post')

    return idx_sequence_pad_list


def data_generator(excel_dir, train_rate=0.8):
    all_data = get_all_excel_data(excel_dir)
    # char_inputs, loc_inputs, char_labels, res_labels = get_train_input_label(all_data)
    # idx_seq_pad_inputs = char_tokenizer(char_inputs, Config.tokenizerPath)
    perm = np.arange(len(all_data))

    np.random.shuffle(perm)

    num_train = int(len(all_data)*train_rate)
    # print(perm)
    # print(num_train, perm[:num_train])
    train_data = np.array(all_data)[perm[:num_train]]
    test_data = np.array(all_data)[perm[num_train:]]
    train_dict = {}
    test_dict = {}

    train_char_inputs, train_loc_inputs, train_char_labels, train_res_labels = get_train_input_label(train_data)
    train_dict['loc_inputs'] = train_loc_inputs
    train_dict['char_labels'] = train_char_labels
    train_dict['res_labels'] = train_res_labels
    train_dict['char_inputs'] = train_char_inputs
    train_char_idx_inputs = char_tokenizer(train_char_inputs, Config.tokenizerPath)
    train_dict['char_idx_inputs'] = train_char_idx_inputs

    test_char_inputs, test_loc_inputs, test_char_labels, test_res_labels = get_train_input_label(test_data)
    test_dict['loc_inputs'] = test_loc_inputs
    test_dict['char_labels'] = test_char_labels
    test_dict['res_labels'] = test_res_labels
    test_dict['char_inputs'] = test_char_inputs
    test_char_idx_inputs = char_tokenizer(test_char_inputs, Config.tokenizerPath)
    test_dict['char_idx_inputs'] = test_char_idx_inputs

    return train_dict, test_dict

def data_generator2(excel_dir, train_rate=0.8):
    all_data = get_all_excel_data(excel_dir)
    # char_inputs, loc_inputs, char_labels, res_labels = get_train_input_label(all_data)
    # idx_seq_pad_inputs = char_tokenizer(char_inputs, Config.tokenizerPath)
    perm = np.arange(len(all_data))

    np.random.shuffle(perm)

    num_train = int(len(all_data)*train_rate)
    # print(perm)
    # print(num_train, perm[:num_train])
    train_data = np.array(all_data)[perm[:num_train]]
    test_data = np.array(all_data)[perm[num_train:]]
    train_dict = {}
    test_dict = {}

    train_char_inputs, train_loc_inputs, train_char_labels, train_res_labels = get_train_input_label2(train_data)
    train_dict['loc_inputs'] = train_loc_inputs
    train_dict['char_labels'] = train_char_labels
    train_dict['res_labels'] = train_res_labels
    train_dict['char_inputs'] = train_char_inputs
    train_char_idx_inputs = char_tokenizer(train_char_inputs, tokenizer=Config.tokenizer)
    train_dict['char_idx_inputs'] = train_char_idx_inputs

    test_char_inputs, test_loc_inputs, test_char_labels, test_res_labels = get_train_input_label2(test_data)
    test_dict['loc_inputs'] = test_loc_inputs
    test_dict['char_labels'] = test_char_labels
    test_dict['res_labels'] = test_res_labels
    test_dict['char_inputs'] = test_char_inputs
    test_char_idx_inputs = char_tokenizer(test_char_inputs, tokenizer=Config.tokenizer)
    test_dict['char_idx_inputs'] = test_char_idx_inputs

    return train_dict, test_dict

def change_ocr(ocr_input):
    alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;!?-/()#*@=+&%><"\' '
    def delete_str_i(ocr_, idx):
        return ocr_[:idx] + ocr_[idx + 1:]

    def duplicate_str_i(ocr_, idx):
        return ocr_[:idx] + ocr_[idx] + ocr_[idx:]

    def replace_str_i(ocr_, idx):
        ocr_list = list(ocr_)
        ocr_list[idx] = random.choice(list(alphabet))
        return ''.join(ocr_list)

    change_type = random.choice([1, 2, 3])
    # 第一种方式：删除字符，模拟ocr漏字
    if change_type == 1:
        ocr_len = len(ocr_input)
        if ocr_len < 10:
            return ocr_input
        delete_num = int(ocr_len / 10)
        delete_num = random.choice(range(1, delete_num+1))
        for i in range(delete_num):
            di = random.choice(range(ocr_len - i))
            ocr_input = delete_str_i(ocr_input, di)
    # 第二种方式：冗余字符，模拟ocr多字
    elif change_type == 2:
        ocr_len = len(ocr_input)
        dup_num = int(ocr_len / 10)
        dup_num = max(dup_num, 2)
        dup_num = random.choice(range(1, dup_num))
        for i in range(dup_num):
            di = random.choice(range(ocr_len + i))
            ocr_input = duplicate_str_i(ocr_input, di)
    # 第三种方式：替换字符，模拟ocr错字
    else:
        ocr_len = len(ocr_input)
        replace_num = int(ocr_len / 10)
        replace_num = max(replace_num, 2)
        replace_num = random.choice(range(1, replace_num))
        for i in range(replace_num):
            ri = random.choice(range(ocr_len))
            ocr_input = replace_str_i(ocr_input, ri)
    return ocr_input

def my_smote(train_dict):
    def calculate_pos_neg_rate(train_dict):
        res_labels = train_dict['res_labels']
        pos_idxs = []
        neg_num = 0
        pos_num = 0
        for i in range(len(res_labels)):
            res_label = res_labels[i]
            if res_label == 1:
                pos_idxs.append(i)
                pos_num += 1
            else:
                neg_num += 1
        return pos_idxs, pos_num, neg_num

    def _neg_ocr(_type, ocr):
        neg_ocr_DICT = {"VESSEL" : ['BY SEA', 'DELIVERY TERM', 'Blue Anchor']}
        match_flag = False
        if _type not in neg_ocr_DICT:
            return match_flag, ocr
        ocr_type_list = neg_ocr_DICT[_type]

        if random.random() < 0.1:
            ocr = random.choice(ocr_type_list)
            match_flag = True
        return match_flag, ocr

    pos_idxs, pos_num, neg_num = calculate_pos_neg_rate(train_dict)
    neg_to_pos = float(neg_num) / pos_num
    print(neg_num, pos_num, neg_to_pos)
    if neg_to_pos < 2.0:
        return train_dict
    smote_times = int(neg_to_pos)

    loc_inputs_list = list(train_dict['loc_inputs'])
    #print(loc_inputs_list)
    char_inputs_list = list(train_dict['char_inputs'])
    char_labels_list = list(train_dict['char_labels'])
    res_labels_list = list(train_dict['res_labels'])

    for pos_i in pos_idxs:
        loc_input_i = train_dict['loc_inputs'][pos_i]
        char_input_i = train_dict['char_inputs'][pos_i]
        char_label_i = train_dict['char_labels'][pos_i]
        res_label_i = train_dict['res_labels'][pos_i]

        # print("========================== ")
        # print("original: ", loc_input_i, char_input_i)

        loc_near_i = []
        for pos_j in pos_idxs:
            if pos_i == pos_j:
                continue
            loc_input_j = train_dict['loc_inputs'][pos_j]
            # 如果出现在原始切片的左上角，就丢弃，因为正常切片通常在右下方
            if loc_input_j[0] < loc_input_i[0] or loc_input_j[1] < loc_input_i[1]:
                continue
            dist = np.linalg.norm(np.array(loc_input_i[:2]) - np.array(loc_input_j[:2]))

            loc_near_i.append([loc_input_j, dist])
        loc_near_i = sorted(loc_near_i, key=lambda loc: loc[-1])
        if not loc_near_i:
            print("怎么一个都没有？")
            continue
        if len(loc_near_i) < smote_times:
            for i in range(smote_times + 1 - len(loc_near_i)):
                loc_near_i.append(loc_near_i[-1])

        # 开始进行smote
        for niloc, dist in loc_near_i[:smote_times-1]:
            new_ni_loc = np.concatenate([[loc_input_i[0] + random.random() * (niloc[0] - loc_input_i[0]),
                          loc_input_i[1] + random.random() * (niloc[1] - loc_input_i[1]),
                          loc_input_i[2] * random.uniform(0.8, 1.2)], loc_input_i[3:]])
            # if random.random() > 0.2:
            #     pre_input, post_input = char_input_i.split('\t')
            #     new_char_input_i = pre_input[0] + pre_input[1:] + '\t' + change_ocr(post_input[:-1]) + post_input[-1]
            # else:
            #     new_char_input_i = char_input_i

            pre_input, post_input = char_input_i.split('\t')

            #print(pre_input, post_input, char_label_i, res_label_i)
            new_char_label_i = char_label_i
            new_res_label_i = res_label_i
            if random.random() > 0.2:
                match_flag, content_ocr = _neg_ocr(pre_input[1:], post_input[:-1])
                new_char_input_i = pre_input[0:] + '\t' + change_ocr(content_ocr) + post_input[-1]

                if match_flag:
                    new_char_label_i = 0
                    new_res_label_i = 0
                    #print("smote: ", new_ni_loc, new_char_input_i, new_char_label_i, new_res_label_i)
            else:
                new_char_input_i = char_input_i


            # print("smote: ", new_ni_loc, new_char_input_i)
            loc_inputs_list.append(new_ni_loc)
            char_inputs_list.append(new_char_input_i)
            char_labels_list.append(new_char_label_i)
            res_labels_list.append(new_res_label_i)



    perm = np.arange(len(loc_inputs_list))
    np.random.shuffle(perm)
    # for ele in loc_inputs_list:
    #     print(len(ele), ele)
    # train_dict['loc_inputs'] = np.reshape(np.array(loc_inputs_list), (len(np.array(loc_inputs_list)), 11))
    train_dict['loc_inputs'] = np.array(loc_inputs_list)
    train_dict['char_inputs'] = np.array(char_inputs_list)
    train_dict['char_labels'] = np.array(char_labels_list)
    train_dict['res_labels'] = np.array(res_labels_list)


    _, pos_num, neg_num = calculate_pos_neg_rate(train_dict)
    neg_to_pos = float(neg_num) / pos_num
    print(neg_num, pos_num, neg_to_pos)
    train_char_idx_inputs = char_tokenizer(train_dict['char_inputs'],  tokenizer=Config.tokenizer)
    train_dict['char_idx_inputs'] = train_char_idx_inputs

    return train_dict


if __name__ == "__main__":
    sentence_list = ['s df\tAS D',
                     'df df \n']

    # token = load_tokenizer('tokenizer/tokenizer.pickle')
    # #idx_sequence_pad_list = char_tokenizer(sentence_list, 'tokenizer/tokenizer_nopunc.pickle', 12)
    # print(token.word_counts)
    # print(len(token.word_index), token.word_index)
    # print(token.word_docs)
    # print(token.index_docs)
    #
    # for k,v in token.word_index.items():
    #     print(k,v)
    excel_dir = '/Users/peng_ji/codeHub/rookieCode/result_train_0804050607_neg'
    train_dict, _ = data_generator2(excel_dir, train_rate=0.8)
    my_smote(train_dict)

