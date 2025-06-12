import torch
import torch.nn as nn

class FGM():
    """ Define the adversarial training method FGM to perturb the model embedding parameters """
    def __init__(self, model, epsilon=0.25):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}
    def attack(self, emb_name='word_embeddings'):

        for name, param in self.model.named_parameters():

            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm

                    param.data.add_(r_at)
    def restore(self, emb_name='word_embeddings'):

        for name, param in self.model.named_parameters():

            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]

        self.backup = {}



class SAN(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(SAN, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        :param src:
        :param src_mask:
        :param src_key_padding_mask:
        :return:
        """
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(src2)
        # apply layer normalization
        src = self.norm(src)
        return src


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=2, dropout=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.dropout(self.fc1(x))
        out1 = self.relu(out1)
        out2 = self.dropout(self.fc2(out1))
        return out2



class MultiPromptLogitSentimentClassificationHead(nn.Module):
    def __init__(self, lm, num_class, num_prompts, pseudo_label_words, target_token_id=-1,
                 merge_behavior='sum_logits'):

        super(MultiPromptLogitSentimentClassificationHead, self).__init__()

        self.num_class = num_class
        self.pseudo_label_words = pseudo_label_words
        self.target_token_id = target_token_id
        self.num_prompts = num_prompts
        self.merge_behavior = merge_behavior
        self.lm = lm
        assert self.target_token_id != -1


    def forward(self, input_ids, attention_mask, token_type_ids):
        target_indexes = torch.nonzero(
            input_ids == self.target_token_id)[:, 1]

        lm_outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # print('lm_out=', lm_outputs)

        real_batch_size = len(input_ids) // self.num_prompts

        temp = target_indexes
        for pos in range(real_batch_size // len(temp)):
            target_indexes = torch.cat((target_indexes, temp), -1)
        # print('target_indexes=', target_indexes)
        outputs = []
        for i in range(real_batch_size):
            scores_batch = []
            for j in range(self.num_prompts):
                if self.merge_behavior == 'sum_logits':
                    logits = lm_outputs.logits[
                        i + real_batch_size * j, target_indexes[i + real_batch_size * j], self.pseudo_label_words[j]]
                    scores_batch.append(logits)

            scores_batch = torch.stack(scores_batch, dim=0)
            scores_batch = torch.sum(scores_batch, dim=0)
            outputs.append(scores_batch)
        outputs = torch.stack(outputs, dim=0)

        return outputs





class MultiPromptSentimentClassificationHead(torch.nn.Module):
    def __init__(self, lm, num_class, num_prompts, target_token_id=-1,
                 merge_behavior='concatenate', perturb_prompts=False):
        super(MultiPromptSentimentClassificationHead, self).__init__()

        self.num_class = num_class
        self.num_prompts = num_prompts
        self.target_token_id = target_token_id
        self.merge_behavior = merge_behavior
        self.perturb_prompts = perturb_prompts
        self.lm = lm
        if self.lm.config.architectures[0].startswith('Bert'):
            # if self.lm is BERT, then mask_token_id should be specified
            assert self.target_token_id != -1
            self.lm_type = 'bert'
        else:
            raise Exception('Unsupported language model type.')

        # print("Detected LM type:", self.lm_type)
        # Linear layer
        if self.merge_behavior == 'concatenate':
            # self.linear = torch.nn.Linear(
            #     self.num_prompts * self.lm.config.hidden_size, self.num_class)
            self.mlp = MLP(input_dim=self.num_prompts * self.lm.config.hidden_size,
                           hidden_dim=256,
                           output_dim=self.num_class)
        elif self.merge_behavior == 'sum':
            self.mlp = MLP(input_dim=self.lm.config.hidden_size,
                           hidden_dim=256,
                           output_dim=self.num_class)
            # self.linear = torch.nn.Linear(
            #     self.lm.config.hidden_size, self.num_class)

    def forward(self, input_ids, attention_mask, token_type_ids):

        lr_inputs_batch = []
        # For BERT, we need to find the token in each input with [MASK]
        target_indexes = torch.nonzero(
            input_ids == self.target_token_id)[:, 1]
        lm_outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             output_hidden_states=True)

        real_batch_size = len(input_ids) // self.num_prompts
        for i in range(real_batch_size):
            lr_input = []
            for j in range(self.num_prompts):
                lr_input.append(
                    lm_outputs["hidden_states"][-1][i + real_batch_size * j][target_indexes[i + real_batch_size * j]])
            if self.num_prompts == 1:
                lr_input = lr_input[0]
            lr_inputs_batch.append(lr_input)

        lr_inputs_batch = torch.stack(lr_inputs_batch)
        # print('lr_input_batch=', lr_inputs_batch)
        # print(lr_inputs_batch.shape)
        outputs = self.mlp(lr_inputs_batch)
        # print('out=', outputs)

        return outputs

