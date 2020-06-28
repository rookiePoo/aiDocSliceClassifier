import os
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from config import Config
from raw_excel_process import get_all_excel_data, get_train_input_label

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        print("tokenizer has been loaded.")
        tokenizer = pickle.load(f)
    return tokenizer

def char_tokenizer(sentence_list, tokenizer_path, maxLen=Config.sequenceMaxLength):
    idx_sequence_pad_list = []
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
    np.random.seed(822)
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



if __name__ == "__main__":
    sentence_list = ['s df\tAS D',
                     'df df \n']

    idx_sequence_pad_list = char_tokenizer(sentence_list, 'tokenizer/tokenizer.pickle', 12)
    print(idx_sequence_pad_list)

