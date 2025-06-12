import random
import re
import DatasetCapsulation as Data
from transformers import BertTokenizer
import torch
import spacy
from spacy.lang.en.tag_map import TAG_MAP
tag_map = {tag: i for i, tag in enumerate(TAG_MAP.keys(), 1)}
tag_map["<pad>"] = 0
nlp = spacy.load('en_core_web_sm')

f = open('./dep_utils/all_vocab_v2.txt', 'r', encoding='gbk')
dep_vocab = f.read().split()
# print(dep_vocab)
dep_vocab_to_ids = {word: i for i, word in enumerate(dep_vocab)}



data_path = "./data/Original_dataset/v2/"
dataset_name_list = ["14lap", "14res", "15res", "16res"]
dataset_type_list = ["train_triplets", "dev_triplets", "test_triplets"]

triplet_pattern = re.compile(r'[(](.*?)[)]', re.S)
aspect_and_opinion_pattern = re.compile(r'[\[](.*?)[]]', re.S)
sentiment_pattern = re.compile(r"['](.*?)[']", re.S)

forward_aspect_query_template = ["[CLS]", "find", "the", "aspect", "terms", "in", "the", "text", ".", "[SEP]"]
forward_opinion_query_template = ["[CLS]", "find", "the", "opinion", "terms", "to", "describe", ".", "[SEP]"]
backward_opinion_query_template = ["[CLS]", "find", "the", "opinion", "terms", "in", "the", "text", ".", "[SEP]"]
backward_aspect_query_template = ["[CLS]", "find", "the", "aspect", "term", "that", "is", ".", "[SEP]"]
aspect_sentiment_query_template = ["[CLS]", "the", "makes", "me", "feel", "[MASK]", ".", "[SEP]"]
sentiment_query_template = ["[CLS]", "the", "is", "and", "makes", "me", "feel", "[MASK]", ".", "[SEP]"]
sentiment_answer_template = ["ok", "good", "bad"]
sentiments_mapping = {
    'POS': 0,
    'NEG': 1,
    'NEU': 2
}



# BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def valid(QA: Data.QueryAndAnswer):
    assert len(QA.forward_asp_query) == len(QA.forward_asp_answer_start) \
           == len(QA.forward_asp_answer_end) == len(QA.forward_asp_query_mask) \
           == len(QA.forward_asp_query_seg)
    for i in range(len(QA.forward_opi_query)):
        assert len(QA.forward_opi_query[i]) == len(QA.forward_opi_answer_start[i]) \
               == len(QA.forward_opi_answer_end[i]) == len(QA.forward_opi_query_mask[i]) \
               == len(QA.forward_opi_query_seg[i])
    assert len(QA.backward_opi_query) == len(QA.backward_opi_answer_start) \
           == len(QA.backward_opi_answer_end) == len(QA.backward_opi_query_mask) \
           == len(QA.backward_opi_query_seg)
    for i in range(len(QA.backward_asp_query)):
        assert len(QA.backward_asp_query[i]) == len(QA.backward_asp_answer_start[i]) \
               == len(QA.backward_asp_answer_end[i]) == len(QA.backward_asp_query_mask[i]) \
               == len(QA.backward_asp_query_seg[i])
    assert len(QA.sentiment_query) == len(QA.sentiment_answer) == len(QA.sentiment_query_mask) \
           == len(QA.sentiment_query_seg)
    for i in range(len(QA.sentiment_query)):
        assert len(QA.sentiment_query[i]) == len(QA.sentiment_query_mask[i]) == len(QA.sentiment_query_seg[i])
    for i in range(len(QA.aspect_sentiment_query)):
        assert len(QA.aspect_sentiment_query[i]) == len(QA.aspect_sentiment_query_mask[i]) == len(QA.aspect_sentiment_query_seg[i])



def ids_to_tokens(input_ids_list):
    if not isinstance(input_ids_list[0], list):
        return tokenizer.convert_ids_to_tokens(input_ids_list)
    token_list = []
    for input_ids in input_ids_list:
        token_list.append(tokenizer.convert_ids_to_tokens(input_ids))
    return token_list


# tokens to ids
def tokens_to_ids(QA_list):
    max_len = 0
    for QA in QA_list:
        QA.forward_asp_query = tokenizer.convert_tokens_to_ids(QA.forward_asp_query)
        if len(QA.forward_asp_query) > max_len:
            max_len = len(QA.forward_asp_query)
        for i in range(len(QA.forward_opi_query)):
            QA.forward_opi_query[i] = tokenizer.convert_tokens_to_ids(QA.forward_opi_query[i])
            if len(QA.forward_opi_query[i]) > max_len:
                max_len = len(QA.forward_opi_query[i])
        QA.backward_opi_query = tokenizer.convert_tokens_to_ids(QA.backward_opi_query)
        if len(QA.backward_opi_query) > max_len:
            max_len = len(QA.backward_opi_query)
        for i in range(len(QA.backward_asp_query)):
            QA.backward_asp_query[i] = tokenizer.convert_tokens_to_ids(QA.backward_asp_query[i])
            if len(QA.backward_asp_query[i]) > max_len:
                max_len = len(QA.backward_asp_query[i])
        for i in range(len(QA.sentiment_query)):
            QA.sentiment_query[i] = tokenizer.convert_tokens_to_ids(QA.sentiment_query[i])
            if len(QA.sentiment_query[i]) > max_len:
                max_len = len(QA.sentiment_query[i])
        for i in range(len(QA.aspect_sentiment_query)):
            QA.aspect_sentiment_query[i] = tokenizer.convert_tokens_to_ids(QA.aspect_sentiment_query[i])
            if len(QA.aspect_sentiment_query[i]) > max_len:
                max_len = len(QA.aspect_sentiment_query[i])
        valid(QA)
    return QA_list, max_len
def list_to_object(dataset_object):
    line = []
    forward_asp_query = []
    forward_opi_query = []
    forward_asp_query_mask = []
    forward_asp_query_seg = []
    forward_opi_query_mask = []
    forward_opi_query_seg = []
    forward_asp_answer_start = []
    forward_asp_answer_end = []
    forward_opi_answer_start = []
    forward_opi_answer_end = []
    backward_asp_query = []
    backward_opi_query = []
    backward_asp_query_mask = []
    backward_asp_query_seg = []
    backward_opi_query_mask = []
    backward_opi_query_seg = []
    backward_asp_answer_start = []
    backward_asp_answer_end = []
    backward_opi_answer_start = []
    backward_opi_answer_end = []
    sentiment_query = []
    sentiment_answer = []
    sentiment_query_mask = []
    sentiment_query_seg = []

    aspect_sentiment_query = []
    aspect_sentiment_query_mask = []
    aspect_sentiment_query_seg = []

    for QA in dataset_object:
        line.append(QA.line)
        forward_asp_query.append(QA.forward_asp_query)
        forward_opi_query.append(QA.forward_opi_query)
        forward_asp_query_mask.append(QA.forward_asp_query_mask)
        forward_asp_query_seg.append(QA.forward_asp_query_seg)
        forward_opi_query_mask.append(QA.forward_opi_query_mask)
        forward_opi_query_seg.append(QA.forward_opi_query_seg)
        forward_asp_answer_start.append(QA.forward_asp_answer_start)
        forward_asp_answer_end.append(QA.forward_asp_answer_end)
        forward_opi_answer_start.append(QA.forward_opi_answer_start)
        forward_opi_answer_end.append(QA.forward_opi_answer_end)
        backward_asp_query.append(QA.backward_asp_query)
        backward_opi_query.append(QA.backward_opi_query)
        backward_asp_query_mask.append(QA.backward_asp_query_mask)
        backward_asp_query_seg.append(QA.backward_asp_query_seg)
        backward_opi_query_mask.append(QA.backward_opi_query_mask)
        backward_opi_query_seg.append(QA.backward_opi_query_seg)
        backward_asp_answer_start.append(QA.backward_asp_answer_start)
        backward_asp_answer_end.append(QA.backward_asp_answer_end)
        backward_opi_answer_start.append(QA.backward_opi_answer_start)
        backward_opi_answer_end.append(QA.backward_opi_answer_end)
        sentiment_query.append(QA.sentiment_query)
        sentiment_answer.append(QA.sentiment_answer)
        sentiment_query_mask.append(QA.sentiment_query_mask)
        sentiment_query_seg.append(QA.sentiment_query_seg)

        aspect_sentiment_query.append(QA.aspect_sentiment_query)
        aspect_sentiment_query_mask.append(QA.aspect_sentiment_query_mask)
        aspect_sentiment_query_seg.append(QA.aspect_sentiment_query_seg)

    return Data.QueryAndAnswer(line=line,
                               forward_asp_query=forward_asp_query,
                               forward_opi_query=forward_opi_query,
                               forward_asp_query_mask=forward_asp_query_mask,
                               forward_asp_query_seg=forward_asp_query_seg,
                               forward_opi_query_mask=forward_opi_query_mask,
                               forward_opi_query_seg=forward_opi_query_seg,
                               forward_asp_answer_start=forward_asp_answer_start,
                               forward_asp_answer_end=forward_asp_answer_end,
                               forward_opi_answer_start=forward_opi_answer_start,
                               forward_opi_answer_end=forward_opi_answer_end,
                               backward_asp_query=backward_asp_query,
                               backward_opi_query=backward_opi_query,
                               backward_asp_query_mask=backward_asp_query_mask,
                               backward_asp_query_seg=backward_asp_query_seg,
                               backward_opi_query_mask=backward_opi_query_mask,
                               backward_opi_query_seg=backward_opi_query_seg,
                               backward_asp_answer_start=backward_asp_answer_start,
                               backward_asp_answer_end=backward_asp_answer_end,
                               backward_opi_answer_start=backward_opi_answer_start,
                               backward_opi_answer_end=backward_opi_answer_end,
                               sentiment_query=sentiment_query,
                               sentiment_answer=sentiment_answer,
                               sentiment_query_mask=sentiment_query_mask,
                               sentiment_query_seg=sentiment_query_seg,
                               aspect_sentiment_query=aspect_sentiment_query,
                               aspect_sentiment_query_mask=aspect_sentiment_query_mask,
                               aspect_sentiment_query_seg=aspect_sentiment_query_seg
                               )

def dataset_align(dataset_object, max_tokens_len, max_aspect_num):
    for dataset_type in dataset_type_list:
        tokenized_QA_list = dataset_object[dataset_type]
        for tokenized_QA in tokenized_QA_list:
            # TODO 词法
            tokenized_QA.forward_asp_pos.extend([0] * (max_tokens_len - len(tokenized_QA.forward_asp_pos)))
            tokenized_QA.backward_opi_pos.extend([0] * (max_tokens_len - len(tokenized_QA.backward_opi_pos)))
            for i in range(len(tokenized_QA.forward_opi_pos)):
                tokenized_QA.forward_opi_pos[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.forward_opi_pos[i])))
            for i in range(len(tokenized_QA.backward_asp_pos)):
                tokenized_QA.backward_asp_pos[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.backward_asp_pos[i])))
            # TODO 语法
            tokenized_QA.forward_asp_dep.extend([0] * (max_tokens_len - len(tokenized_QA.forward_asp_dep)))
            tokenized_QA.backward_opi_dep.extend([0] * (max_tokens_len - len(tokenized_QA.backward_opi_dep)))
            for i in range(len(tokenized_QA.forward_opi_dep)):
                tokenized_QA.forward_opi_dep[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.forward_opi_dep[i])))
            for i in range(len(tokenized_QA.backward_asp_dep)):
                tokenized_QA.backward_asp_dep[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.backward_asp_dep[i])))
            # ***************************


            tokenized_QA.forward_asp_query.extend([0] * (max_tokens_len - len(tokenized_QA.forward_asp_query)))
            tokenized_QA.forward_asp_query_mask.extend(
                [0] * (max_tokens_len - len(tokenized_QA.forward_asp_query_mask)))
            tokenized_QA.forward_asp_query_seg.extend(
                [1] * (max_tokens_len - len(tokenized_QA.forward_asp_query_seg)))

            tokenized_QA.forward_asp_answer_start.extend(
                [-1] * (max_tokens_len - len(tokenized_QA.forward_asp_answer_start)))
            tokenized_QA.forward_asp_answer_end.extend(
                [-1] * (max_tokens_len - len(tokenized_QA.forward_asp_answer_end)))

            for i in range(len(tokenized_QA.forward_opi_query)):
                tokenized_QA.forward_opi_query[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.forward_opi_query[i])))
                tokenized_QA.forward_opi_answer_start[i].extend(
                    [-1] * (max_tokens_len - len(tokenized_QA.forward_opi_answer_start[i])))
                tokenized_QA.forward_opi_answer_end[i].extend(
                    [-1] * (max_tokens_len - len(tokenized_QA.forward_opi_answer_end[i])))

                tokenized_QA.forward_opi_query_mask[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.forward_opi_query_mask[i])))
                tokenized_QA.forward_opi_query_seg[i].extend(
                    [1] * (max_tokens_len - len(tokenized_QA.forward_opi_query_seg[i])))

            tokenized_QA.backward_opi_query.extend([0] * (max_tokens_len - len(tokenized_QA.backward_opi_query)))

            tokenized_QA.backward_opi_query_mask.extend(
                [0] * (max_tokens_len - len(tokenized_QA.backward_opi_query_mask)))
            tokenized_QA.backward_opi_query_seg.extend(
                [1] * (max_tokens_len - len(tokenized_QA.backward_opi_query_seg)))

            tokenized_QA.backward_opi_answer_start.extend(
                [-1] * (max_tokens_len - len(tokenized_QA.backward_opi_answer_start)))
            tokenized_QA.backward_opi_answer_end.extend(
                [-1] * (max_tokens_len - len(tokenized_QA.backward_opi_answer_end)))

            for i in range(len(tokenized_QA.backward_asp_query)):
                tokenized_QA.backward_asp_query[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.backward_asp_query[i])))
                tokenized_QA.backward_asp_answer_start[i].extend(
                    [-1] * (max_tokens_len - len(tokenized_QA.backward_asp_answer_start[i])))
                tokenized_QA.backward_asp_answer_end[i].extend(
                    [-1] * (max_tokens_len - len(tokenized_QA.backward_asp_answer_end[i])))

                tokenized_QA.backward_asp_query_mask[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.backward_asp_query_mask[i])))
                tokenized_QA.backward_asp_query_seg[i].extend(
                    [1] * (max_tokens_len - len(tokenized_QA.backward_asp_query_seg[i])))

            for i in range(len(tokenized_QA.sentiment_query)):
                tokenized_QA.sentiment_query[i].extend([0] * (max_tokens_len - len(tokenized_QA.sentiment_query[i])))

                tokenized_QA.sentiment_query_mask[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.sentiment_query_mask[i])))
                tokenized_QA.sentiment_query_seg[i].extend(
                    [1] * (max_tokens_len - len(tokenized_QA.sentiment_query_seg[i])))

            for i in range(len(tokenized_QA.aspect_sentiment_query)):
                tokenized_QA.aspect_sentiment_query[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.aspect_sentiment_query[i])))
                tokenized_QA.aspect_sentiment_query_mask[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.aspect_sentiment_query_mask[i])))
                tokenized_QA.aspect_sentiment_query_seg[i].extend(
                    [1] * (max_tokens_len - len(tokenized_QA.aspect_sentiment_query_seg[i])))

            for i in range(max_aspect_num - len(tokenized_QA.forward_opi_query)):
                # TODO new_add
                tokenized_QA.forward_opi_pos.insert(-1, tokenized_QA.forward_opi_pos[0])
                tokenized_QA.backward_asp_pos.insert(-1, tokenized_QA.backward_asp_pos[0])

                tokenized_QA.forward_opi_dep.insert(-1, tokenized_QA.forward_opi_dep[0])
                tokenized_QA.backward_asp_dep.insert(-1, tokenized_QA.backward_asp_dep[0])
                # ***************************

                tokenized_QA.forward_opi_query.insert(-1, tokenized_QA.forward_opi_query[0])
                tokenized_QA.forward_opi_query_mask.insert(-1, tokenized_QA.forward_opi_query_mask[0])
                tokenized_QA.forward_opi_query_seg.insert(-1, tokenized_QA.forward_opi_query_seg[0])

                tokenized_QA.forward_opi_answer_start.insert(-1, tokenized_QA.forward_opi_answer_start[0])
                tokenized_QA.forward_opi_answer_end.insert(-1, tokenized_QA.forward_opi_answer_end[0])

                tokenized_QA.backward_asp_query.insert(-1, tokenized_QA.backward_asp_query[0])
                tokenized_QA.backward_asp_query_mask.insert(-1, tokenized_QA.backward_asp_query_mask[0])
                tokenized_QA.backward_asp_query_seg.insert(-1, tokenized_QA.backward_asp_query_seg[0])

                tokenized_QA.backward_asp_answer_start.insert(-1, tokenized_QA.backward_asp_answer_start[0])
                tokenized_QA.backward_asp_answer_end.insert(-1, tokenized_QA.backward_asp_answer_end[0])

                tokenized_QA.sentiment_query.insert(-1, tokenized_QA.sentiment_query[0])
                tokenized_QA.sentiment_query_mask.insert(-1, tokenized_QA.sentiment_query_mask[0])
                tokenized_QA.sentiment_query_seg.insert(-1, tokenized_QA.sentiment_query_seg[0])

                tokenized_QA.sentiment_answer.insert(-1, tokenized_QA.sentiment_answer[0])

                tokenized_QA.aspect_sentiment_query.insert(-1, tokenized_QA.aspect_sentiment_query[0])
                tokenized_QA.aspect_sentiment_query_mask.insert(-1, tokenized_QA.aspect_sentiment_query_mask[0])
                tokenized_QA.aspect_sentiment_query_seg.insert(-1, tokenized_QA.aspect_sentiment_query_seg[0])

            valid(tokenized_QA)
            # if random.random() > 0.99:
            #     print_QA(tokenized_QA)
    return dataset_object



def get_start_end(str_list):
    index_list = [int(s) for s in str_list]
    for i in range(1, len(index_list)):
        assert index_list[i] - index_list[i - 1] == 1
    return [index_list[0], index_list[-1]]

def make_QA(line, word_list, aspect_list, opinion_list, sentiment_list, pos_ids_list, dep_list):
    # word_list.append("[SEP]")
    forward_asp_query = forward_aspect_query_template + word_list
    forward_asp_query_mask = [1] * len(forward_asp_query)
    forward_asp_query_seg = [0] * len(forward_aspect_query_template) + [1] * len(word_list)
    forward_asp_answer_start = [-1] * len(forward_aspect_query_template) + [0] * len(word_list)
    forward_asp_answer_end = [-1] * len(forward_aspect_query_template) + [0] * len(word_list)
    forward_opi_query = []
    forward_opi_query_mask = []
    forward_opi_query_seg = []
    forward_opi_answer_start = []
    forward_opi_answer_end = []

    backward_opi_query = backward_opinion_query_template + word_list
    backward_opi_query_mask = [1] * len(backward_opi_query)
    backward_opi_query_seg = [0] * len(backward_opinion_query_template) + [1] * len(word_list)
    backward_opi_answer_start = [-1] * len(backward_opinion_query_template) + [0] * len(word_list)
    backward_opi_answer_end = [-1] * len(backward_opinion_query_template) + [0] * len(word_list)
    backward_asp_query = []
    backward_asp_query_mask = []
    backward_asp_query_seg = []
    backward_asp_answer_start = []
    backward_asp_answer_end = []

    sentiment_query = []
    sentiment_query_mask = []
    sentiment_query_seg = []
    sentiment_answer = sentiment_list

    aspect_sentiment_query = []
    aspect_sentiment_query_mask = []
    aspect_sentiment_query_seg = []
    # TODO POS
    forward_asp_pos = [tag_map["<pad>"]] * len(forward_asp_query)
    forward_asp_pos[
        len(forward_aspect_query_template):len(forward_aspect_query_template) + len(pos_ids_list) + 1] = pos_ids_list
    backward_opi_pos = [tag_map["<pad>"]] * len(backward_opi_query)
    backward_opi_pos[
        len(backward_opinion_query_template):len(backward_opinion_query_template) + len(pos_ids_list) + 1] = pos_ids_list
    forward_opi_pos = []
    backward_asp_pos = []
    #TODO Dep
    forward_asp_dep = [0] * len(forward_asp_query)
    forward_asp_dep[
        len(forward_aspect_query_template):len(forward_aspect_query_template) + len(dep_list) + 1] = dep_list

    backward_opi_dep = [0] * len(backward_opi_query)
    backward_opi_dep[
        len(backward_opinion_query_template):len(backward_opinion_query_template) + len(dep_list) + 1] = dep_list

    forward_opi_dep = []
    backward_asp_dep = []


    for i in range(len(aspect_list)):
        asp = aspect_list[i]
        opi = opinion_list[i]

        forward_asp_answer_start[len(forward_aspect_query_template) + asp[0]] = 1
        forward_asp_answer_end[len(forward_aspect_query_template) + asp[1]] = 1
        # TODO

        opi_query_temp = forward_opinion_query_template[0:7] + word_list[asp[0]:asp[1] + 1] + \
                         forward_opinion_query_template[7:] + word_list
        # print('opi_query_temp=', opi_query_temp)

        # TODO forward_opi pos
        length = 7 + len(word_list[asp[0]:asp[1] + 1]) + len(forward_opinion_query_template[7:])
        opi_pos_temp = [tag_map["<pad>"]] * len(opi_query_temp)
        opi_pos_temp[length:length + len(pos_ids_list) + 1] = pos_ids_list
        forward_opi_pos.append(opi_pos_temp)

        opi_dep_temp = [0] * len(opi_query_temp)
        opi_dep_temp[length:length + len(dep_list) + 1] = dep_list
        forward_opi_dep.append(opi_dep_temp)
        # ************************

        forward_opi_query.append(opi_query_temp)

        opi_query_mask_temp = [1] * len(opi_query_temp)
        opi_query_seg_temp = [0] * (len(opi_query_temp) - len(word_list)) + [1] * len(word_list)
        forward_opi_query_mask.append(opi_query_mask_temp)
        forward_opi_query_seg.append(opi_query_seg_temp)



        opi_answer_start_temp = [-1] * (len(opi_query_temp) - len(word_list)) + [0] * len(word_list)
        opi_answer_start_temp[len(opi_query_temp) - len(word_list) + opi[0]] = 1
        opi_answer_end_temp = [-1] * (len(opi_query_temp) - len(word_list)) + [0] * len(word_list)
        opi_answer_end_temp[len(opi_query_temp) - len(word_list) + opi[1]] = 1
        forward_opi_answer_start.append(opi_answer_start_temp)
        forward_opi_answer_end.append(opi_answer_end_temp)

        backward_opi_answer_start[len(backward_opinion_query_template) + opi[0]] = 1
        backward_opi_answer_end[len(backward_opinion_query_template) + opi[1]] = 1
        # TODO

        asp_query_temp = backward_aspect_query_template[0:7] + word_list[opi[0]:opi[1] + 1] + \
                         backward_aspect_query_template[7:] + word_list
        # print('asp_query_temp=', asp_query_temp)
        backward_asp_query.append(asp_query_temp)

        # TODO backward_asp pos
        length = 7 + len(word_list[opi[0]:opi[1] + 1]) + len(backward_aspect_query_template[7:])
        asp_pos_temp = [tag_map["<pad>"]] * len(asp_query_temp)
        asp_pos_temp[length:length + len(pos_ids_list) + 1] = pos_ids_list
        backward_asp_pos.append(asp_pos_temp)

        asp_dep_temp = [0] * len(asp_query_temp)
        asp_dep_temp[length:length + len(dep_list) + 1] = dep_list
        backward_asp_dep.append(asp_dep_temp)
        # ************************

        asp_query_mask_temp = [1] * len(asp_query_temp)
        asp_query_seg_temp = [0] * (len(asp_query_temp) - len(word_list)) + [1] * len(word_list)
        backward_asp_query_mask.append(asp_query_mask_temp)
        backward_asp_query_seg.append(asp_query_seg_temp)

        asp_answer_start_temp = [-1] * (len(asp_query_temp) - len(word_list)) + [0] * len(word_list)
        asp_answer_start_temp[len(asp_query_temp) - len(word_list) + asp[0]] = 1
        asp_answer_end_temp = [-1] * (len(asp_query_temp) - len(word_list)) + [0] * len(word_list)
        asp_answer_end_temp[len(asp_query_temp) - len(word_list) + asp[1]] = 1
        backward_asp_answer_start.append(asp_answer_start_temp)
        backward_asp_answer_end.append(asp_answer_end_temp)
        # TODO
        sentiment_query_temp = sentiment_query_template[0:2] + word_list[asp[0]:asp[1] + 1] + \
                               sentiment_query_template[2:3] + word_list[opi[0]:opi[1] + 1] + \
                               sentiment_query_template[3:] + word_list
        # print('sentiment_query_temp=', sentiment_query_temp)

        sentiment_query.append(sentiment_query_temp)
        sentiment_query_mask_temp = [1] * len(sentiment_query_temp)
        sentiment_query_seg_temp = [0] * (len(sentiment_query_temp) - len(word_list)) + [1] * len(word_list)
        sentiment_query_mask.append(sentiment_query_mask_temp)
        sentiment_query_seg.append(sentiment_query_seg_temp)
        # TODO A-S
        aspect_sentiment_query_temp = aspect_sentiment_query_template[0:2] + word_list[asp[0]:asp[1] + 1] + \
                                      aspect_sentiment_query_template[2:] + word_list
        # print('spect_sentiment_query_temp=', aspect_sentiment_query_temp)
        aspect_sentiment_query.append(aspect_sentiment_query_temp)
        aspect_sentiment_query_mask_temp = [1] * len(aspect_sentiment_query_temp)
        aspect_sentiment_query_seg_temp = [0] * (len(aspect_sentiment_query_temp) - len(word_list)) + [1] * len(word_list)
        aspect_sentiment_query_mask.append(aspect_sentiment_query_mask_temp)
        aspect_sentiment_query_seg.append(aspect_sentiment_query_seg_temp)



    return Data.QueryAndAnswer(line=line,
                               forward_asp_query=forward_asp_query,
                               forward_opi_query=forward_opi_query,
                               forward_asp_query_mask=forward_asp_query_mask,
                               forward_asp_query_seg=forward_asp_query_seg,
                               forward_opi_query_mask=forward_opi_query_mask,
                               forward_opi_query_seg=forward_opi_query_seg,
                               forward_asp_answer_start=forward_asp_answer_start,
                               forward_asp_answer_end=forward_asp_answer_end,
                               forward_opi_answer_start=forward_opi_answer_start,
                               forward_opi_answer_end=forward_opi_answer_end,
                               backward_asp_query=backward_asp_query,
                               backward_opi_query=backward_opi_query,
                               backward_asp_query_mask=backward_asp_query_mask,
                               backward_asp_query_seg=backward_asp_query_seg,
                               backward_opi_query_mask=backward_opi_query_mask,
                               backward_opi_query_seg=backward_opi_query_seg,
                               backward_asp_answer_start=backward_asp_answer_start,
                               backward_asp_answer_end=backward_asp_answer_end,
                               backward_opi_answer_start=backward_opi_answer_start,
                               backward_opi_answer_end=backward_opi_answer_end,
                               sentiment_query=sentiment_query,
                               sentiment_answer=sentiment_answer,
                               sentiment_query_mask=sentiment_query_mask,
                               sentiment_query_seg=sentiment_query_seg,
                               aspect_sentiment_query=aspect_sentiment_query,
                               aspect_sentiment_query_mask=aspect_sentiment_query_mask,
                               aspect_sentiment_query_seg=aspect_sentiment_query_seg,
                               forward_asp_pos=forward_asp_pos, forward_opi_pos=forward_opi_pos,
                               backward_asp_pos=backward_asp_pos, backward_opi_pos=backward_opi_pos,
                               forward_asp_dep=forward_asp_dep, forward_opi_dep=forward_opi_dep,
                               backward_asp_dep=backward_asp_dep, backward_opi_dep=backward_opi_dep
                               )

def encode_solve(word_list, aspect_list, opinion_list):
    end = -1
    index = []
    new_word_list = []
    new_aspect_list = []
    new_opinion_list = []
    new_pos_list = []
    new_dep_list = []

    for word in word_list:
        doc = nlp(word)
        spacy_words = [item.text for item in doc]
        spacy_pos = [item.tag_ for item in doc]
        encode_words = tokenizer.convert_ids_to_tokens(tokenizer.encode(word))
        encode_words_len = len(encode_words)
        idx = 0
        for i in range(1, encode_words_len - 1):
            new_word_list.append(encode_words[i])
            new_pos_list.append(spacy_pos[idx])
            new_dep_list.append(dep_vocab_to_ids[spacy_words[idx]])
            if spacy_words[idx] == encode_words[i]:
                idx += 1
        start = end + 1
        end = start + encode_words_len - 3
        index.append([start, end])
    assert len(new_pos_list) == len(new_word_list) == len(new_dep_list)
    new_pos_ids_list = [tag_map[item] for item in new_pos_list]
    # print('new_word_list=', new_word_list)
    # print('new_pos_ids_list=', new_pos_ids_list)
    # print('new_dep_list=', new_dep_list)

    # print(new_pos_ids_list)
    for i in range(len(aspect_list)):
        new_aspect_list.append([index[aspect_list[i][0]][0], index[aspect_list[i][1]][1]])
        new_opinion_list.append([index[opinion_list[i][0]][0], index[opinion_list[i][1]][1]])
    return new_word_list, new_aspect_list, new_opinion_list, new_pos_ids_list, new_dep_list


def line_data_process(line, isQA=True):
    # Line sample:
    # It is easy to use , fast and has great graphics for the money .####[([10], [9], 'POS'), ([4], [2], 'POS')]
    split = line.split("####")
    # print(split)
    assert len(split) == 2
    max_aspect_num = 0
    max_len = 0
    word_list = split[0].split()
    word_list = [word.lower() for word in word_list]
    triplet_str_list = re.findall(triplet_pattern, split[1])
    aspect_list = [re.findall(aspect_and_opinion_pattern, triplet)[0] for triplet in triplet_str_list]
    aspect_list = [get_start_end(aspect.split(',')) for aspect in aspect_list]
    if len(aspect_list) > max_aspect_num:
        max_aspect_num = len(aspect_list)
    opinion_list = [re.findall(aspect_and_opinion_pattern, triplet)[1] for triplet in triplet_str_list]
    opinion_list = [get_start_end(opinion.split(',')) for opinion in opinion_list]
    sentiment_list = [sentiments_mapping[re.findall(sentiment_pattern, triplet)[0]] for triplet in triplet_str_list]
    assert len(aspect_list) > 0 and len(opinion_list) > 0 and len(sentiment_list) > 0
    assert len(aspect_list) == len(opinion_list) == len(sentiment_list)
    # TODO
    word_list, aspect_list, opinion_list, pos_ids_list, dep_list = encode_solve(word_list, aspect_list, opinion_list)
    for i in range(len(aspect_list)):
        if (aspect_list[i][1] - aspect_list[i][0] + 1) > max_len:
            max_len = aspect_list[i][1] - aspect_list[i][0] + 1
        if (opinion_list[i][1] - opinion_list[i][0] + 1) > max_len:
            max_len = opinion_list[i][1] - opinion_list[i][0] + 1
    if isQA:
        return make_QA(line, word_list, aspect_list, opinion_list, sentiment_list, pos_ids_list, dep_list), max_aspect_num, max_len
    else:
        return word_list, aspect_list, opinion_list, sentiment_list, pos_ids_list, dep_list


def train_data_process(text):
    QA_list = []
    max_aspect_num = 0
    max_len = 0
    for line in text:
        QA, max_aspect_temp, max_len_temp = line_data_process(line)
        if max_aspect_temp > max_aspect_num:
            max_aspect_num = max_aspect_temp
        if max_len_temp > max_len:
            max_len = max_len_temp
        QA_list.append(QA)
    return QA_list, max_aspect_num, max_len
def test_data_process(text):
    test_dataset = []
    for line in text:
        word_list, aspect_list_temp, opinion_list_temp, sentiment_list_temp, pos_ids_list, dep_list = line_data_process(line, isQA=False)
        aspect_list = []
        opinion_list = []
        asp_opi_list = []
        asp_sent_list = []
        triplet_list = []
        for i in range(0, len(aspect_list_temp)):
            if aspect_list_temp[i] not in aspect_list:
                aspect_list.append(aspect_list_temp[i])
            if opinion_list_temp[i] not in opinion_list:
                opinion_list.append(opinion_list_temp[i])
            asp_opi_temp = [aspect_list_temp[i][0], aspect_list_temp[i][1], opinion_list_temp[i][0],
                            opinion_list_temp[i][1]]
            asp_sent_temp = [aspect_list_temp[i][0], aspect_list_temp[i][1], sentiment_list_temp[i]]
            triplet_temp = [aspect_list_temp[i][0], aspect_list_temp[i][1], opinion_list_temp[i][0],
                            opinion_list_temp[i][1], sentiment_list_temp[i]]
            asp_opi_list.append(asp_opi_temp)
            if asp_sent_temp not in asp_sent_list:
                asp_sent_list.append(asp_sent_temp)
            triplet_list.append(triplet_temp)
        test_dataset.append(Data.TestDataset(
            word_list=word_list,
            aspect_list=aspect_list,
            opinion_list=opinion_list,
            asp_opi_list=asp_opi_list,
            asp_sent_list=asp_sent_list,
            triplet_list=triplet_list,
            pos_ids_list=pos_ids_list,
            dep_list=dep_list
        ))
    return test_dataset


def dataset_process():
    for dataset_name in dataset_name_list:
        train_output_path = "./data/preprocess/v2/" + dataset_name + ".pt"
        test_output_path = "./data/preprocess/v2/" + dataset_name + "_test.pt"
        train_dataset_object = {}
        test_dataset_object = {}
        max_tokens_len = 0
        max_aspect_num = 0
        max_len = 0
        for dataset_type in dataset_type_list:
            file = open(data_path + dataset_name + "/" + dataset_type + ".txt", "r", encoding="utf-8")
            text_lines = file.readlines()
            QA_list, max_aspect_temp, max_len_temp = train_data_process(text_lines)
            train_dataset_object[dataset_type], max_tokens_temp = tokens_to_ids(QA_list)
            test_dataset_object[dataset_type] = test_data_process(text_lines)
            if max_tokens_temp > max_tokens_len:
                max_tokens_len = max_tokens_temp
            if max_aspect_temp > max_aspect_num:
                max_aspect_num = max_aspect_temp
            if max_len_temp > max_len:
                max_len = max_len_temp
        train_dataset_object = dataset_align(train_dataset_object, max_tokens_len, max_aspect_num)
        train_dataset_object['max_tokens_len'] = max_tokens_len
        train_dataset_object['max_aspect_num'] = max_aspect_num
        train_dataset_object['max_len'] = max_len
        print(max_len)
        torch.save(train_dataset_object, train_output_path)
        torch.save(test_dataset_object, test_output_path)


if __name__ == '__main__':
    dataset_process()
