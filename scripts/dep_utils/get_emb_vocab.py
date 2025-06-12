import numpy as np
import torch
import spacy
nlp = spacy.load('en_core_web_sm')

data_path = "../data/Original_dataset/v2/"
dataset_name_list = ["14lap", "14rest", "15rest", "16rest"]
if 'v2' in data_path:
    dataset_name_list = ["14lap", "14res", "15res", "16res"]
dataset_type_list = ["train", "dev", "test"]
# dataset_name_list = ["14lap"]
# dataset_type_list = ["dev"]
def dataset_process():
    all_vocab = ['<pad>']
    all_emb = [[0.] * 96]
    for dataset_name in dataset_name_list:
        for dataset_type in dataset_type_list:
            if 'v1' in data_path:
                path = data_path + dataset_name + "/" + dataset_type + ".txt"
            elif 'v2' in data_path:
                path = data_path + dataset_name + "/" + dataset_type + "_triplets.txt"
            text_file = open(path, "r", encoding="utf-8")
            text_lines = text_file.readlines()
            for line in text_lines:
                split = line.split("####")
                assert len(split) == 2
                word_list = split[0].split()
                word_list = [word.lower() for word in word_list]
                for word in word_list:
                    doc = nlp(word)
                    for item in doc:
                        if item.text not in all_vocab:
                            all_vocab.append(item.text)
                            all_emb.append(item.vector)
            print('{} !'.format(path))
    assert len(all_emb) == len(all_vocab)
    print(len(all_emb))
    if 'v1' in data_path:
        np.save('all_emb.npy', all_emb)
        with open('all_vocab.txt', mode='w') as f:
            for word in all_vocab:
                f.write(word + '\n')
    elif 'v2' in data_path:
        np.save('all_emb_v2.npy', all_emb)
        with open('all_vocab_v2.txt', mode='w') as f:
            for word in all_vocab:
                f.write(word + '\n')



if __name__ == '__main__':
    dataset_process()
