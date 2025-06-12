from transformers import BertModel, BertForMaskedLM, BertTokenizer
import torch.nn as nn
import torch
import numpy as np
from Layers import MultiPromptSentimentClassificationHead
from DataProcessV1 import tag_map

sentiment_prompts = [{"prompt": "The {aspect} is {opinion} and makes me feel [MASK].", "labels": ["ok", "good", "bad"]}]
aspect_sentiment_prompts = [{"prompt": "The {aspect} made me feel [MASK].", "labels": ["ok", "good", "bad"]}]

def get_sentiment_word_ids(tokenizer, sentiment_prompts):
    sentiment_word_ids = []
    for sp in sentiment_prompts:
        sentiment_word_ids.append(
            [tokenizer.convert_tokens_to_ids(w) for w in sp['labels']])

    return sentiment_word_ids

def create_emb_layer(weights_matrix, trainable=True):
    tensor_emb = torch.FloatTensor(weights_matrix)
    num_embeddings, embedding_dim = tensor_emb.shape
    emb_layer = nn.Embedding.from_pretrained(tensor_emb)
    emb_layer.weight.requires_grad = trainable

    return emb_layer, num_embeddings, embedding_dim



class BERTModel(nn.Module):
    def __init__(self, hidden_size, bert_model_type, pos_embedding_size=32, dependency_emb=True):

        super(BERTModel, self).__init__()

        if bert_model_type == './bert_base_uncased':

            self._bert = BertForMaskedLM.from_pretrained(bert_model_type)
            self._tokenizer = BertTokenizer.from_pretrained(bert_model_type)
            print('bert-base-uncased model loaded')

        else:
            raise KeyError('bert_model_type should be bert-based-uncased.')
        last_hidden_size = hidden_size
        self.dependency_emb = dependency_emb
        self.pos_embedding_size = pos_embedding_size

        # TODO lexical dependencies
        if self.pos_embedding_size > 0:
            self.embeddings = nn.Embedding(len(tag_map), pos_embedding_size)

            last_hidden_size += pos_embedding_size
        # TODO Syntax Dependency
        if dependency_emb:
            dep_matrix = np.load('./dep_utils/all_emb.npy')
            self.dep_embeddings, _, embedding_dim = create_emb_layer(dep_matrix)

            last_hidden_size += embedding_dim
        # print('dep_emd=', embedding_dim)
        # print('pos_emd=', pos_embedding_size)
        # print('hidden_size=', last_hidden_size)


        self.classifier_a_ao_start = nn.Linear(last_hidden_size, 2)
        self.classifier_a_ao_end = nn.Linear(last_hidden_size, 2)
        self.classifier_o_oa_start = nn.Linear(last_hidden_size, 2)
        self.classifier_o_oa_end = nn.Linear(last_hidden_size, 2)

        self.dropout = nn.Dropout(0.1)

        self.classifier_sentiment = MultiPromptSentimentClassificationHead(
            lm=self._bert, num_class=3, num_prompts=1,
            target_token_id=self._tokenizer.mask_token_id)

    def forward(self, query_tensor, query_mask, query_seg, step, pos_ids=None, dep_ids=None):

        hidden_states = self._bert(input_ids=query_tensor, attention_mask=query_mask, token_type_ids=query_seg,
                                   output_hidden_states=True).hidden_states[-1]
        if step == 'A' or step == 'AO':
            if self.pos_embedding_size > 0 and pos_ids is not None:
                pos_output = self.embeddings(pos_ids)

                hidden_states = torch.cat((hidden_states, pos_output), dim=-1)

            if self.dependency_emb and dep_ids is not None:
                dep_output = self.dep_embeddings(dep_ids)
                hidden_states = torch.cat([hidden_states, dep_output], dim=-1)

            predict_start = self.classifier_a_ao_start(self.dropout(hidden_states))
            predict_end = self.classifier_a_ao_end(self.dropout(hidden_states))
            return predict_start, predict_end

        elif step == 'O' or step == 'OA':
            if self.pos_embedding_size > 0 and pos_ids is not None:
                pos_output = self.embeddings(pos_ids)
                hidden_states = torch.cat((hidden_states, pos_output), dim=-1)

            if self.dependency_emb and dep_ids is not None:
                dep_output = self.dep_embeddings(dep_ids)
                hidden_states = torch.cat([hidden_states, dep_output], dim=-1)

            predict_start = self.classifier_o_oa_start(self.dropout(hidden_states))
            predict_end = self.classifier_o_oa_end(self.dropout(hidden_states))
            return predict_start, predict_end

        elif step == 'S' or step == 'AS':
            return self.classifier_sentiment(input_ids=query_tensor, attention_mask=query_mask,
                                             token_type_ids=query_seg)
        else:
            raise KeyError('step error.')
