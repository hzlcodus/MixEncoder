from transformers import BertModel, BertForSequenceClassification, BertConfig
from transformers.models.bert.out_vector_modeling_bert import MyBertModel, DecoderLayerChunk
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import torch.nn.functional
import torch.nn.functional as F
from nlp_model import ParallelEncoderConfig, QAClassifier

from my_function import get_rep_by_avg, dot_attention, clean_input_ids, clean_input_embeddings
try:
    from apex import amp
    APEX_FLAG = True
except:
    APEX_FLAG = False

# used for 1-1 classification task
class UltraClassifyParallelEncoder(nn.Module):
    def __init__(self, config: ParallelEncoderConfig):

        super(UltraClassifyParallelEncoder, self).__init__()

        self.config = config

        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        self.this_bert_config = BertConfig.from_pretrained(config.pretrained_bert_path)

        # contain the parameters of encoder
        self.bert_model = MyBertModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)
        self.embeddings = self.bert_model.get_input_embeddings()

        # create models for decoder
        self.num_attention_heads = self.this_bert_config.num_attention_heads
        self.attention_head_size = int(self.this_bert_config.hidden_size / self.this_bert_config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # contain cls for extract context vectors
        self.context_cls = nn.Parameter(torch.nn.init.xavier_normal_(torch.randn((self.config.context_num, self.all_head_size))))
        self.composition_layer = ContextLayer(self.this_bert_config.hidden_size, config.context_num)

        self.decoder = nn.ModuleDict({
            # used by candidate context embeddings to enrich themselves
            'candidate_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            'candidate_key': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                            range(self.this_bert_config.num_hidden_layers)]),
            'candidate_value': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            # used to project representations into same space
            'layer_chunks': nn.ModuleList(
                [DecoderLayerChunk(self.this_bert_config) for _ in range(self.this_bert_config.num_hidden_layers)]),
            # used to compress candidate context embeddings into a vector
            'candidate_composition_layer': SelectLastLayer(),
            # used to generate query to compress Text_A thus get its representation
            'compress_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                             range(self.this_bert_config.num_hidden_layers)]),
            # used to update hidden states with self-att output as input
            'LSTM': MyLSTMBlock(self.this_bert_config.hidden_size),
        })

        self.mean_layer = MeanLayer()

        # create classifier
        self.classifier = QAClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)

        # control which layer use decoder
        self.enrich_flag = [False] * self.this_bert_config.num_hidden_layers
        # self.enrich_flag[0] = True
        self.enrich_flag[-1] = True
        # for i in range(6, 12):
        #     self.enrich_flag[i] = True
        print("*"*50)
        print(f"Enrich Layer: {self.enrich_flag}")
        print("*"*50)

        self.init_decoder_params()

        # remove no use component
        for index, flag in enumerate(self.enrich_flag):
            if not flag:
                for key in ['layer_chunks', 'candidate_query', 'compress_query', 'candidate_key', 'candidate_value']:
                    self.decoder[key][index] = None

    # use parameters of pre-trained encoder to initialize decoder
    def init_decoder_params(self):
        #  read parameters for layer chunks from encoder
        for index in range(self.this_bert_config.num_hidden_layers):
            this_static = self.decoder['layer_chunks'][index].state_dict()
            pretrained_static = self.bert_model.encoder.layer[index].state_dict()

            updating_layer_static = {}
            for k in this_static.keys():
                new_k = k.replace('attention.', 'attention.output.', 1)
                updating_layer_static[k] = pretrained_static[new_k]

            this_static.update(updating_layer_static)
            self.decoder['layer_chunks'][index].load_state_dict(this_static)

        #  read parameters for query from encoder
        pretrained_static = self.bert_model.state_dict()
        query_static = self.decoder['candidate_query'].state_dict()

        updating_query_static = {}
        for k in query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        query_static.update(updating_query_static)
        self.decoder['candidate_query'].load_state_dict(query_static)

        #  read parameters for compress_query from encoder
        compress_query_static = self.decoder['compress_query'].state_dict()

        updating_query_static = {}
        for k in compress_query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        compress_query_static.update(updating_query_static)
        self.decoder['compress_query'].load_state_dict(compress_query_static)

        #  read parameters for key from encoder
        key_static = self.decoder['candidate_key'].state_dict()

        updating_key_static = {}
        for k in key_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.key.' + k.split(".")[1]
            updating_key_static[k] = pretrained_static[full_key]

        key_static.update(updating_key_static)
        self.decoder['candidate_key'].load_state_dict(key_static)

        #  read parameters for value from encoder
        value_static = self.decoder['candidate_value'].state_dict()

        updating_value_static = {}
        for k in value_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.value.' + k.split(".")[1]
            updating_value_static[k] = pretrained_static[full_key]

        value_static.update(updating_value_static)
        self.decoder['candidate_value'].load_state_dict(value_static)

    def prepare_for_cls_encoding(self, input_ids, token_type_ids, attention_mask):
        # (all_candidate_num, max_seq_len)
        candidate_seq_len = input_ids.shape[-1]

        # remove cls
        input_ids = input_ids[:, 1:]
        token_type_ids = token_type_ids[:, 1:]
        attention_mask = attention_mask[:, 1:]

        # remove padding
        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        # remove last token in order to prepend context_cls
        if candidate_seq_len > self.config.text_max_len + 1 - self.config.context_num:
            remove_len = candidate_seq_len - (self.config.text_max_len + 1 - self.config.context_num)
            input_ids = input_ids[:, :-remove_len]
            token_type_ids = token_type_ids[:, :-remove_len]
            attention_mask = attention_mask[:, :-remove_len]

        # prepend context_cls
        input_embeds = self.embeddings(input_ids)
        expanded_context_cls = self.context_cls.unsqueeze(0).repeat(input_ids.shape[0], 1, 1)

        # get embedding
        input_embeds = torch.cat((expanded_context_cls, input_embeds), dim=1)

        # process mask and type id
        temp_zeros = torch.zeros((token_type_ids.shape[0], self.config.context_num),
                                 device=input_embeds.device, dtype=torch.int)
        temp_ones = torch.ones((attention_mask.shape[0], self.config.context_num),
                               device=input_embeds.device, dtype=torch.int)
        token_type_ids = torch.cat((temp_zeros, token_type_ids), dim=1)
        attention_mask = torch.cat((temp_ones, attention_mask), dim=1)

        return input_embeds, token_type_ids, attention_mask

    def prepare_candidates(self, input_ids, token_type_ids, attention_mask):
        """ Pre-compute representations. Return shape: (-1, context_num, context_dim) """

        input_embeds, token_type_ids, attention_mask =  self.prepare_for_cls_encoding(input_ids, token_type_ids, attention_mask)

        out = self.bert_model(inputs_embeds=input_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask,
                              enrich_candidate_by_question=[False]*self.this_bert_config.num_hidden_layers)
        last_hidden_state = out['last_hidden_state']

        candidate_embeddings = last_hidden_state[:, :self.config.context_num, :]

        cls_candidate_embeddings = self.composition_layer(last_hidden_state[:, self.config.context_num:, :],
                                                          attention_mask=attention_mask[:, self.config.context_num:])

        return (candidate_embeddings + cls_candidate_embeddings)/2.0

    def do_queries_classify(self, input_ids, token_type_ids, attention_mask, candidate_context_embeddings, **kwargs):
        """ use pre-compute candidate to get logits """
        no_aggregator = kwargs.get('no_aggregator', False)
        no_enricher = kwargs.get('no_enricher', False)

        # (batch size, context num, embedding len)
        candidate_context_embeddings = candidate_context_embeddings.reshape(input_ids.shape[0], self.config.context_num,
                                                                            candidate_context_embeddings.shape[-1])

        lstm_initial_state = torch.zeros(candidate_context_embeddings.shape[0], 1, candidate_context_embeddings.shape[-1],
                                         device=candidate_context_embeddings.device)

        # (query_num, context num + 1, dim)
        candidate_input_embeddings = torch.cat((candidate_context_embeddings, lstm_initial_state), dim=1)

        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        a_out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                enrich_candidate_by_question=self.enrich_flag, candidate_embeddings=candidate_input_embeddings,
                                decoder=self.decoder, candidate_num=1)

        a_last_hidden_state, decoder_output = a_out['last_hidden_state'], a_out['decoder_output']

        b_context_embeddings = decoder_output[:, :-1, :].reshape(decoder_output.shape[0], 1,
                                                                 self.config.context_num,
                                                                 decoder_output.shape[-1])

        # control ablation
        if no_aggregator:
            # (query_num, 1, dim)
            a_embeddings = a_last_hidden_state[:, 0, :].unsqueeze(1)
        else:
            # (query_num, candidate_num, dim)
            a_embeddings = decoder_output[:, -1:, :]

        if no_enricher:
            b_embeddings = self.mean_layer(candidate_context_embeddings[:, :-1, :])
        else:
            # (query_num, candidate_num, dim)
            b_embeddings = self.mean_layer(b_context_embeddings[:, :, :-1, :]).squeeze(-2)

        logits = self.classifier(a_embedding=a_embeddings, b_embedding=b_embeddings).squeeze(1)

        return logits

    # 如果不把a传进q，就会退化成最普通的QAModel，已经被验证过了
    # b_text can be pre-computed
    def forward(self, a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, **kwargs):

        # must be clarified for training
        no_aggregator = kwargs.get('no_aggregator')
        no_enricher = kwargs.get('no_enricher')

        b_embeddings = self.prepare_candidates(input_ids=b_input_ids,
                                               attention_mask=b_attention_mask,
                                               token_type_ids=b_token_type_ids)

        logits = self.do_queries_classify(input_ids=a_input_ids, attention_mask=a_attention_mask,
                                          token_type_ids=a_token_type_ids,
                                          candidate_context_embeddings=b_embeddings,
                                          no_aggregator=no_aggregator,
                                          no_enricher=no_enricher)

        return logits


# used for 1-1 classification task
class DisenClassifyParallelEncoder(nn.Module):
    def __init__(self, config: ParallelEncoderConfig):

        super(DisenClassifyParallelEncoder, self).__init__()

        self.config = config

        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        self.this_bert_config = BertConfig.from_pretrained(config.pretrained_bert_path)

        # contain the parameters of encoder
        self.bert_model = MyBertModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)

        # self.embeddings = self.bert_model.get_input_embeddings()

        # used to compressing candidate to generate context vectors
        self.composition_layer = ContextLayer(self.this_bert_config.hidden_size, config.context_num)

        # create models for decoder
        self.num_attention_heads = self.this_bert_config.num_attention_heads
        self.attention_head_size = int(self.this_bert_config.hidden_size / self.this_bert_config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.decoder = nn.ModuleDict({
            # used by candidate context embeddings to enrich themselves
            'candidate_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            'candidate_key': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                            range(self.this_bert_config.num_hidden_layers)]),
            'candidate_value': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            # used to project representations into same space
            'layer_chunks': nn.ModuleList(
                [DecoderLayerChunk(self.this_bert_config) for _ in range(self.this_bert_config.num_hidden_layers)]),
            # used to compress candidate context embeddings into a vector
            'candidate_composition_layer': SelectLastLayer(),
            # used to generate query to compress Text_A thus get its representation
            'compress_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                             range(self.this_bert_config.num_hidden_layers)]),
            # used to update hidden states with self-att output as input
            'LSTM': MyLSTMBlock(self.this_bert_config.hidden_size),
        })

        self.mean_layer = MeanLayer()

        # create classifier
        self.classifier = QAClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)

        # control which layer use decoder
        self.enrich_flag = [False] * self.this_bert_config.num_hidden_layers
        # self.enrich_flag[0] = True
        self.enrich_flag[-1] = True
        print("*"*50)
        print(f"Enrich Layer: {self.enrich_flag}")
        print("*"*50)

        self.init_decoder_params()

        # remove no use component
        for index, flag in enumerate(self.enrich_flag):
            if not flag:
                for key in ['layer_chunks', 'candidate_query', 'compress_query', 'candidate_key', 'candidate_value']:
                    self.decoder[key][index] = None

    # use parameters of pre-trained encoder to initialize decoder
    def init_decoder_params(self):
        #  read parameters for layer chunks from encoder
        for index in range(self.this_bert_config.num_hidden_layers):
            this_static = self.decoder['layer_chunks'][index].state_dict()
            pretrained_static = self.bert_model.encoder.layer[index].state_dict()

            updating_layer_static = {}
            for k in this_static.keys():
                new_k = k.replace('attention.', 'attention.output.', 1)
                updating_layer_static[k] = pretrained_static[new_k]

            this_static.update(updating_layer_static)
            self.decoder['layer_chunks'][index].load_state_dict(this_static)

        #  read parameters for query from encoder
        pretrained_static = self.bert_model.state_dict()
        query_static = self.decoder['candidate_query'].state_dict()

        updating_query_static = {}
        for k in query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        query_static.update(updating_query_static)
        self.decoder['candidate_query'].load_state_dict(query_static)

        #  read parameters for compress_query from encoder
        compress_query_static = self.decoder['compress_query'].state_dict()

        updating_query_static = {}
        for k in compress_query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        compress_query_static.update(updating_query_static)
        self.decoder['compress_query'].load_state_dict(compress_query_static)

        #  read parameters for key from encoder
        key_static = self.decoder['candidate_key'].state_dict()

        updating_key_static = {}
        for k in key_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.key.' + k.split(".")[1]
            updating_key_static[k] = pretrained_static[full_key]

        key_static.update(updating_key_static)
        self.decoder['candidate_key'].load_state_dict(key_static)

        #  read parameters for value from encoder
        value_static = self.decoder['candidate_value'].state_dict()

        updating_value_static = {}
        for k in value_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.value.' + k.split(".")[1]
            updating_value_static[k] = pretrained_static[full_key]

        value_static.update(updating_value_static)
        self.decoder['candidate_value'].load_state_dict(value_static)

    def prepare_candidates(self, input_ids, token_type_ids, attention_mask):
        """ Pre-compute representations. """

        # reshape and encode candidates
        # (all_candidate_num, dim)
        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                              enrich_candidate_by_question=[False]*self.this_bert_config.num_hidden_layers)
        last_hidden_state = out['last_hidden_state']

        # get (all_candidate_num, context_num, dim)
        candidate_embeddings = self.composition_layer(last_hidden_state, attention_mask=attention_mask)

        return candidate_embeddings

    def do_queries_classify(self, input_ids, token_type_ids, attention_mask, candidate_context_embeddings, **kwargs):
        """ use pre-compute candidate to get logits """
        no_aggregator = kwargs.get('no_aggregator', False)
        no_enricher = kwargs.get('no_enricher', False)

        candidate_context_embeddings = candidate_context_embeddings.reshape(input_ids.shape[0], self.config.context_num,
                                                                            candidate_context_embeddings.shape[-1])

        lstm_initial_state = torch.zeros(candidate_context_embeddings.shape[0], 1, candidate_context_embeddings.shape[-1],
                                         device=candidate_context_embeddings.device)

        # (query_num, candidate_context_num + candidate_num, dim)
        candidate_input_embeddings = torch.cat((candidate_context_embeddings, lstm_initial_state), dim=1)

        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        a_out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                enrich_candidate_by_question=self.enrich_flag, candidate_embeddings=candidate_input_embeddings,
                                decoder=self.decoder, candidate_num=1)

        a_last_hidden_state, decoder_output = a_out['last_hidden_state'], a_out['decoder_output']

        b_context_embeddings = decoder_output[:, :-1, :].reshape(decoder_output.shape[0], 1,
                                                                 self.config.context_num,
                                                                 decoder_output.shape[-1])

        # control ablation
        if no_aggregator:
            # (query_num, 1, dim)
            a_embeddings = a_last_hidden_state[:, 0, :].unsqueeze(1)
        else:
            # (query_num, candidate_num, dim)
            a_embeddings = decoder_output[:, -1:, :]

        if no_enricher:
            if self.config.context_num == 1:
                b_embeddings = candidate_context_embeddings
            else:
                b_embeddings = self.mean_layer(candidate_context_embeddings[:, :-1, :])
        else:
            # (query_num, candidate_num, dim)
            if self.config.context_num == 1:
                b_embeddings = b_context_embeddings.squeeze(-2)
            else:
                b_embeddings = self.mean_layer(b_context_embeddings[:, :, :-1, :]).squeeze(-2)

        logits = self.classifier(a_embedding=a_embeddings, b_embedding=b_embeddings).squeeze(1)

        return logits

    # 如果不把a传进q，就会退化成最普通的QAModel，已经被验证过了
    # b_text can be pre-computed
    def forward(self, a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, **kwargs):
        # must be clarified for training
        no_aggregator = kwargs.get('no_aggregator')
        no_enricher = kwargs.get('no_enricher')

        b_embeddings = self.prepare_candidates(input_ids=b_input_ids,
                                               attention_mask=b_attention_mask,
                                               token_type_ids=b_token_type_ids)

        logits = self.do_queries_classify(input_ids=a_input_ids, attention_mask=a_attention_mask,
                                          token_type_ids=a_token_type_ids,
                                          candidate_context_embeddings=b_embeddings,
                                          no_aggregator=no_aggregator,
                                          no_enricher=no_enricher)

        return logits


# used for 1-1 classification task
class CLSClassifyParallelEncoder(nn.Module):
    def __init__(self, config: ParallelEncoderConfig):

        super(CLSClassifyParallelEncoder, self).__init__()

        self.config = config

        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        self.this_bert_config = BertConfig.from_pretrained(config.pretrained_bert_path)

        # contain the parameters of encoder
        self.bert_model = MyBertModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)
        self.embeddings = self.bert_model.get_input_embeddings()

        # create models for decoder
        self.num_attention_heads = self.this_bert_config.num_attention_heads
        self.attention_head_size = int(self.this_bert_config.hidden_size / self.this_bert_config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # contain cls for extract context vectors
        self.context_cls = nn.Parameter(torch.nn.init.xavier_normal_(torch.randn((self.config.context_num, self.all_head_size))))

        self.decoder = nn.ModuleDict({
            # used by candidate context embeddings to enrich themselves
            'candidate_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            'candidate_key': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                            range(self.this_bert_config.num_hidden_layers)]),
            'candidate_value': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            # used to project representations into same space
            'layer_chunks': nn.ModuleList(
                [DecoderLayerChunk(self.this_bert_config) for _ in range(self.this_bert_config.num_hidden_layers)]),
            # used to compress candidate context embeddings into a vector
            'candidate_composition_layer': MeanLayer(),
            # used to generate query to compress Text_A thus get its representation
            'compress_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                             range(self.this_bert_config.num_hidden_layers)]),
            # used to update hidden states with self-att output as input
            'LSTM': MyLSTMBlock(self.this_bert_config.hidden_size),
        })

        # create classifier
        self.classifier = QAClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)

        # control which layer use decoder
        self.enrich_flag = [False] * self.this_bert_config.num_hidden_layers
        # self.enrich_flag[0] = True
        self.enrich_flag[-1] = True
        # for i in range(6, 12):
        #     self.enrich_flag[i] = True
        print("*"*50)
        print(f"Enrich Layer: {self.enrich_flag}")
        print("*"*50)

        self.init_decoder_params()

        # remove no use component
        for index, flag in enumerate(self.enrich_flag):
            if not flag:
                for key in ['layer_chunks', 'candidate_query', 'compress_query', 'candidate_key', 'candidate_value']:
                    self.decoder[key][index] = None

    # use parameters of pre-trained encoder to initialize decoder
    def init_decoder_params(self):
        #  read parameters for layer chunks from encoder
        for index in range(self.this_bert_config.num_hidden_layers):
            this_static = self.decoder['layer_chunks'][index].state_dict()
            pretrained_static = self.bert_model.encoder.layer[index].state_dict()

            updating_layer_static = {}
            for k in this_static.keys():
                new_k = k.replace('attention.', 'attention.output.', 1)
                updating_layer_static[k] = pretrained_static[new_k]

            this_static.update(updating_layer_static)
            self.decoder['layer_chunks'][index].load_state_dict(this_static)

        #  read parameters for query from encoder
        pretrained_static = self.bert_model.state_dict()
        query_static = self.decoder['candidate_query'].state_dict()

        updating_query_static = {}
        for k in query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        query_static.update(updating_query_static)
        self.decoder['candidate_query'].load_state_dict(query_static)

        #  read parameters for compress_query from encoder
        compress_query_static = self.decoder['compress_query'].state_dict()

        updating_query_static = {}
        for k in compress_query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        compress_query_static.update(updating_query_static)
        self.decoder['compress_query'].load_state_dict(compress_query_static)

        #  read parameters for key from encoder
        key_static = self.decoder['candidate_key'].state_dict()

        updating_key_static = {}
        for k in key_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.key.' + k.split(".")[1]
            updating_key_static[k] = pretrained_static[full_key]

        key_static.update(updating_key_static)
        self.decoder['candidate_key'].load_state_dict(key_static)

        #  read parameters for value from encoder
        value_static = self.decoder['candidate_value'].state_dict()

        updating_value_static = {}
        for k in value_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.value.' + k.split(".")[1]
            updating_value_static[k] = pretrained_static[full_key]

        value_static.update(updating_value_static)
        self.decoder['candidate_value'].load_state_dict(value_static)

    def prepare_for_cls_encoding(self, input_ids, token_type_ids, attention_mask):
        # (all_candidate_num, max_seq_len)
        candidate_seq_len = input_ids.shape[-1]

        # remove cls
        input_ids = input_ids[:, 1:]
        token_type_ids = token_type_ids[:, 1:]
        attention_mask = attention_mask[:, 1:]

        # remove padding
        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        # remove last token in order to prepend context_cls
        if candidate_seq_len > self.config.text_max_len + 1 - self.config.context_num:
            remove_len = candidate_seq_len - (self.config.text_max_len + 1 - self.config.context_num)
            input_ids = input_ids[:, :-remove_len]
            token_type_ids = token_type_ids[:, :-remove_len]
            attention_mask = attention_mask[:, :-remove_len]

        # prepend context_cls
        input_embeds = self.embeddings(input_ids)
        expanded_context_cls = self.context_cls.unsqueeze(0).repeat(input_ids.shape[0], 1, 1)

        # get embedding
        input_embeds = torch.cat((expanded_context_cls, input_embeds), dim=1)

        # process mask and type id
        temp_zeros = torch.zeros((token_type_ids.shape[0], self.config.context_num),
                                 device=input_embeds.device, dtype=torch.int)
        temp_ones = torch.ones((attention_mask.shape[0], self.config.context_num),
                               device=input_embeds.device, dtype=torch.int)
        token_type_ids = torch.cat((temp_zeros, token_type_ids), dim=1)
        attention_mask = torch.cat((temp_ones, attention_mask), dim=1)

        return input_embeds, token_type_ids, attention_mask

    def prepare_candidates(self, input_ids, token_type_ids, attention_mask):
        """ Pre-compute representations. Return shape: (-1, context_num, context_dim) """

        input_embeds, token_type_ids, attention_mask =  self.prepare_for_cls_encoding(input_ids, token_type_ids, attention_mask)

        out = self.bert_model(inputs_embeds=input_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask,
                              enrich_candidate_by_question=[False]*self.this_bert_config.num_hidden_layers)
        last_hidden_state = out['last_hidden_state']

        candidate_embeddings = last_hidden_state[:, :self.config.context_num, :]
        return candidate_embeddings

    def do_queries_classify(self, input_ids, token_type_ids, attention_mask, candidate_context_embeddings, **kwargs):
        """ use pre-compute candidate to get logits """
        no_aggregator = kwargs.get('no_aggregator', False)
        no_enricher = kwargs.get('no_enricher', False)

        # (batch size, context num, embedding len)
        candidate_context_embeddings = candidate_context_embeddings.reshape(input_ids.shape[0], self.config.context_num,
                                                                            candidate_context_embeddings.shape[-1])

        lstm_initial_state = torch.zeros(candidate_context_embeddings.shape[0], 1, candidate_context_embeddings.shape[-1],
                                         device=candidate_context_embeddings.device)

        # (query_num, context num + 1, dim)
        candidate_input_embeddings = torch.cat((candidate_context_embeddings, lstm_initial_state), dim=1)

        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        a_out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                enrich_candidate_by_question=self.enrich_flag, candidate_embeddings=candidate_input_embeddings,
                                decoder=self.decoder, candidate_num=1)

        a_last_hidden_state, decoder_output = a_out['last_hidden_state'], a_out['decoder_output']

        b_context_embeddings = decoder_output[:, :-1, :].reshape(decoder_output.shape[0], 1,
                                                                 self.config.context_num,
                                                                 decoder_output.shape[-1])

        # control ablation
        if no_aggregator:
            # (query_num, 1, dim)
            a_embeddings = a_last_hidden_state[:, 0, :].unsqueeze(1)
        else:
            # (query_num, candidate_num, dim)
            a_embeddings = decoder_output[:, -1:, :]

        if no_enricher:
            b_embeddings = self.decoder['candidate_composition_layer'](candidate_context_embeddings)
        else:
            # (query_num, candidate_num, dim)
            b_embeddings = self.decoder['candidate_composition_layer'](b_context_embeddings).squeeze(-2)

        logits = self.classifier(a_embedding=a_embeddings, b_embedding=b_embeddings).squeeze(1)

        return logits

    # 如果不把a传进q，就会退化成最普通的QAModel，已经被验证过了
    # b_text can be pre-computed
    def forward(self, a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, **kwargs):

        # must be clarified for training
        no_aggregator = kwargs.get('no_aggregator')
        no_enricher = kwargs.get('no_enricher')

        b_embeddings = self.prepare_candidates(input_ids=b_input_ids,
                                               attention_mask=b_attention_mask,
                                               token_type_ids=b_token_type_ids)

        logits = self.do_queries_classify(input_ids=a_input_ids, attention_mask=a_attention_mask,
                                          token_type_ids=a_token_type_ids,
                                          candidate_context_embeddings=b_embeddings,
                                          no_aggregator=no_aggregator,
                                          no_enricher=no_enricher)

        return logits


# used for 1-1 classification task
class DisenCLSClassifyParallelEncoder(nn.Module):
    def __init__(self, config: ParallelEncoderConfig):

        super(DisenCLSClassifyParallelEncoder, self).__init__()

        self.config = config

        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        self.this_bert_config = BertConfig.from_pretrained(config.pretrained_bert_path)

        # contain the parameters of encoder
        self.bert_model = MyBertModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)
        self.embeddings = self.bert_model.get_input_embeddings()

        # create models for decoder
        self.num_attention_heads = self.this_bert_config.num_attention_heads
        self.attention_head_size = int(self.this_bert_config.hidden_size / self.this_bert_config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # contain cls for extract context vectors
        self.context_cls = nn.Parameter(torch.nn.init.xavier_normal_(torch.randn((self.config.context_num, self.all_head_size))))

        self.decoder = nn.ModuleDict({
            # used by candidate context embeddings to enrich themselves
            'candidate_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            'candidate_key': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                            range(self.this_bert_config.num_hidden_layers)]),
            'candidate_value': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            # used to project representations into same space
            'layer_chunks': nn.ModuleList(
                [DecoderLayerChunk(self.this_bert_config) for _ in range(self.this_bert_config.num_hidden_layers)]),
            # used to compress candidate context embeddings into a vector
            'candidate_composition_layer': SelectLastLayer(),
            # used to generate query to compress Text_A thus get its representation
            'compress_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                             range(self.this_bert_config.num_hidden_layers)]),
            # used to update hidden states with self-att output as input
            'LSTM': MyLSTMBlock(self.this_bert_config.hidden_size),
        })

        # create classifier
        self.classifier = QAClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)

        self.mean_layer = MeanLayer()

        # control which layer use decoder
        self.enrich_flag = [False] * self.this_bert_config.num_hidden_layers
                
        if self.config.used_layers is None:
            self.enrich_flag[-1] = True
        elif self.config.used_layers == "pass":
            pass
        elif ',' in self.config.used_layers:
            layer_indices = [eval(temp_i) for temp_i in self.config.used_layers.split(",")]
            for l_i in layer_indices:
                self.enrich_flag[l_i] = True
        elif '-' in self.config.used_layers:
            layer_indices = [eval(temp_i) for temp_i in self.config.used_layers.split("-")]
            
            if len(layer_indices) != 2:
                raise Exception(f"You should offer used_layers as x-x, your given arg is {self.config.used_layers}.")
            
            for l_i in range(layer_indices[0], layer_indices[1] + 1):
                self.enrich_flag[l_i] = True
        else:
            self.enrich_flag[eval(self.config.used_layers)] = True

        # if last interaction layer is not the last Transformer layer, we can exit early
        self.last_layer_index = -1
        for temp_i, flag in enumerate(self.enrich_flag):
            if flag:
                self.last_layer_index = temp_i
        
        print("*" * 50)
        print(f"Enrich Layer: {self.enrich_flag}")
        print("*" * 50)
        
        self.init_decoder_params()

        # remove no use component
        for index, flag in enumerate(self.enrich_flag):
            if not flag:
                for key in ['layer_chunks', 'candidate_query', 'compress_query', 'candidate_key', 'candidate_value']:
                    self.decoder[key][index] = None

        print(f"Context Num: {self.config.context_num}, Note that 1 context will be used as hint!!!")
        print("*" * 50)

    # use parameters of pre-trained encoder to initialize decoder
    def init_decoder_params(self):
        #  read parameters for layer chunks from encoder
        for index in range(self.this_bert_config.num_hidden_layers):
            this_static = self.decoder['layer_chunks'][index].state_dict()
            pretrained_static = self.bert_model.encoder.layer[index].state_dict()

            updating_layer_static = {}
            for k in this_static.keys():
                new_k = k.replace('attention.', 'attention.output.', 1)
                updating_layer_static[k] = pretrained_static[new_k]

            this_static.update(updating_layer_static)
            self.decoder['layer_chunks'][index].load_state_dict(this_static)

        #  read parameters for query from encoder
        pretrained_static = self.bert_model.state_dict()
        query_static = self.decoder['candidate_query'].state_dict()

        updating_query_static = {}
        for k in query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        query_static.update(updating_query_static)
        self.decoder['candidate_query'].load_state_dict(query_static)

        #  read parameters for compress_query from encoder
        compress_query_static = self.decoder['compress_query'].state_dict()

        updating_query_static = {}
        for k in compress_query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        compress_query_static.update(updating_query_static)
        self.decoder['compress_query'].load_state_dict(compress_query_static)

        #  read parameters for key from encoder
        key_static = self.decoder['candidate_key'].state_dict()

        updating_key_static = {}
        for k in key_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.key.' + k.split(".")[1]
            updating_key_static[k] = pretrained_static[full_key]

        key_static.update(updating_key_static)
        self.decoder['candidate_key'].load_state_dict(key_static)

        #  read parameters for value from encoder
        value_static = self.decoder['candidate_value'].state_dict()

        updating_value_static = {}
        for k in value_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.value.' + k.split(".")[1]
            updating_value_static[k] = pretrained_static[full_key]

        value_static.update(updating_value_static)
        self.decoder['candidate_value'].load_state_dict(value_static)

    def prepare_for_cls_encoding(self, input_ids, token_type_ids, attention_mask):
        # (all_candidate_num, max_seq_len)
        candidate_seq_len = input_ids.shape[-1]

        # remove cls
        input_ids = input_ids[:, 1:]
        token_type_ids = token_type_ids[:, 1:]
        attention_mask = attention_mask[:, 1:]

        # remove padding
        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        # remove last token in order to prepend context_cls
        if candidate_seq_len > self.config.text_max_len + 1 - self.config.context_num:
            remove_len = candidate_seq_len - (self.config.text_max_len + 1 - self.config.context_num)
            input_ids = input_ids[:, :-remove_len]
            token_type_ids = token_type_ids[:, :-remove_len]
            attention_mask = attention_mask[:, :-remove_len]

        # prepend context_cls
        input_embeds = self.embeddings(input_ids)
        expanded_context_cls = self.context_cls.unsqueeze(0).repeat(input_ids.shape[0], 1, 1)

        # get embedding
        input_embeds = torch.cat((expanded_context_cls, input_embeds), dim=1)

        # process mask and type id
        temp_zeros = torch.zeros((token_type_ids.shape[0], self.config.context_num),
                                 device=input_embeds.device, dtype=torch.int)
        temp_ones = torch.ones((attention_mask.shape[0], self.config.context_num),
                               device=input_embeds.device, dtype=torch.int)
        token_type_ids = torch.cat((temp_zeros, token_type_ids), dim=1)
        attention_mask = torch.cat((temp_ones, attention_mask), dim=1)

        return input_embeds, token_type_ids, attention_mask

    def prepare_candidates(self, input_ids, token_type_ids, attention_mask):
        """ Pre-compute representations. Return shape: (-1, context_num, context_dim) """

        input_embeds, token_type_ids, attention_mask =  self.prepare_for_cls_encoding(input_ids, token_type_ids, attention_mask)

        out = self.bert_model(inputs_embeds=input_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask,
                              enrich_candidate_by_question=[False]*self.this_bert_config.num_hidden_layers)
        last_hidden_state = out['last_hidden_state']

        candidate_embeddings = last_hidden_state[:, :self.config.context_num, :]
        return candidate_embeddings

    def do_queries_classify(self, input_ids, token_type_ids, attention_mask, candidate_context_embeddings, **kwargs):
        """ use pre-compute candidate to get logits """
        no_aggregator = kwargs.get('no_aggregator', False)
        no_enricher = kwargs.get('no_enricher', False)

        # (batch size, context num, embedding len)
        candidate_context_embeddings = candidate_context_embeddings.reshape(input_ids.shape[0], self.config.context_num,
                                                                            candidate_context_embeddings.shape[-1])

        lstm_initial_state = torch.zeros(candidate_context_embeddings.shape[0], 1, candidate_context_embeddings.shape[-1],
                                         device=candidate_context_embeddings.device)

        # (query_num, context num + 1, dim)
        candidate_input_embeddings = torch.cat((candidate_context_embeddings, lstm_initial_state), dim=1)

        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        a_out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                enrich_candidate_by_question=self.enrich_flag, candidate_embeddings=candidate_input_embeddings,
                                decoder=self.decoder, candidate_num=1, last_layer_index=self.last_layer_index)

        a_last_hidden_state, decoder_output = a_out['last_hidden_state'], a_out['decoder_output']

        b_context_embeddings = decoder_output[:, :-1, :].reshape(decoder_output.shape[0], 1,
                                                                 self.config.context_num,
                                                                 decoder_output.shape[-1])

        # control ablation
        if no_aggregator:
            # (query_num, 1, dim)
            a_embeddings = a_last_hidden_state[:, 0, :].unsqueeze(1)
        else:
            # (query_num, candidate_num, dim)
            a_embeddings = decoder_output[:, -1:, :]

        if no_enricher:
            if self.config.context_num == 1:
                b_embeddings = candidate_context_embeddings
            else:
                b_embeddings = self.mean_layer(candidate_context_embeddings[:, :-1, :])
        else:
            # (query_num, candidate_num, dim)
            if self.config.context_num == 1:
                b_embeddings = b_context_embeddings.squeeze(-2)
            else:
                b_embeddings = self.mean_layer(b_context_embeddings[:, :, :-1, :]).squeeze(-2)

        logits = self.classifier(a_embedding=a_embeddings, b_embedding=b_embeddings).squeeze(1)

        return logits

    # 如果不把a传进q，就会退化成最普通的QAModel，已经被验证过了
    # b_text can be pre-computed
    def forward(self, a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, **kwargs):

        # must be clarified for training
        no_aggregator = kwargs.get('no_aggregator')
        no_enricher = kwargs.get('no_enricher')

        b_embeddings = self.prepare_candidates(input_ids=b_input_ids,
                                               attention_mask=b_attention_mask,
                                               token_type_ids=b_token_type_ids)

        logits = self.do_queries_classify(input_ids=a_input_ids, attention_mask=a_attention_mask,
                                          token_type_ids=a_token_type_ids,
                                          candidate_context_embeddings=b_embeddings,
                                          no_aggregator=no_aggregator,
                                          no_enricher=no_enricher)

        return logits


# used for 1-n match task
class CLSMatchParallelEncoder(nn.Module):
    def __init__(self, config: ParallelEncoderConfig):

        super(CLSMatchParallelEncoder, self).__init__()

        self.config = config

        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        self.this_bert_config = BertConfig.from_pretrained(config.pretrained_bert_path)

        # contain the parameters of encoder
        self.bert_model = MyBertModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)
        self.embeddings = self.bert_model.get_input_embeddings()

        # info
        self.num_attention_heads = self.this_bert_config.num_attention_heads
        self.attention_head_size = int(self.this_bert_config.hidden_size / self.this_bert_config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # contain cls for extract context vectors
        self.context_cls = nn.Parameter(torch.nn.init.xavier_normal_(torch.randn((self.config.context_num, self.all_head_size))))

        # create models for decoder
        self.decoder = nn.ModuleDict({
            # used by candidate context embeddings to enrich themselves
            'candidate_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            'candidate_key': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                            range(self.this_bert_config.num_hidden_layers)]),
            'candidate_value': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            # used to project representations into same space
            'layer_chunks': nn.ModuleList(
                [DecoderLayerChunk(self.this_bert_config) for _ in range(self.this_bert_config.num_hidden_layers)]),
            # used to compress candidate context embeddings into a vector
            'candidate_composition_layer': MeanLayer(),
            # used to generate query to compress Text_A thus get its representation
            'compress_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                             range(self.this_bert_config.num_hidden_layers)]),
            # used to update hidden states with self-att output as input
            'LSTM': MyLSTMBlock(self.this_bert_config.hidden_size),
        })

        # control which layer use decoder
        self.enrich_flag = [False] * self.this_bert_config.num_hidden_layers
        # self.enrich_flag[0] = True
        self.enrich_flag[-1] = True
        print("*" * 50)
        print(f"Enrich Layer: {self.enrich_flag}")
        print("*" * 50)

        self.init_decoder_params()

        # remove no use component
        for index, flag in enumerate(self.enrich_flag):
            if not flag:
                for key in ['layer_chunks', 'candidate_query', 'compress_query', 'candidate_key', 'candidate_value']:
                    self.decoder[key][index] = None

    # use parameters (layer chunk, q, q_1, k, v) of a pre-trained encoder to initialize the decoder
    def init_decoder_params(self):
        #  read parameters for layer chunks from encoder
        for index in range(self.this_bert_config.num_hidden_layers):
            this_static = self.decoder['layer_chunks'][index].state_dict()
            pretrained_static = self.bert_model.encoder.layer[index].state_dict()

            updating_layer_static = {}
            for k in this_static.keys():
                new_k = k.replace('attention.', 'attention.output.', 1)
                updating_layer_static[k] = pretrained_static[new_k]

            this_static.update(updating_layer_static)
            self.decoder['layer_chunks'][index].load_state_dict(this_static)

        #  read parameters for query from encoder
        pretrained_static = self.bert_model.state_dict()
        query_static = self.decoder['candidate_query'].state_dict()

        updating_query_static = {}
        for k in query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        query_static.update(updating_query_static)
        self.decoder['candidate_query'].load_state_dict(query_static)

        #  read parameters for compress_query from encoder
        compress_query_static = self.decoder['compress_query'].state_dict()

        updating_query_static = {}
        for k in compress_query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        compress_query_static.update(updating_query_static)
        self.decoder['compress_query'].load_state_dict(compress_query_static)

        #  read parameters for key from encoder
        key_static = self.decoder['candidate_key'].state_dict()

        updating_key_static = {}
        for k in key_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.key.' + k.split(".")[1]
            updating_key_static[k] = pretrained_static[full_key]

        key_static.update(updating_key_static)
        self.decoder['candidate_key'].load_state_dict(key_static)

        #  read parameters for value from encoder
        value_static = self.decoder['candidate_value'].state_dict()

        updating_value_static = {}
        for k in value_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.value.' + k.split(".")[1]
            updating_value_static[k] = pretrained_static[full_key]

        value_static.update(updating_value_static)
        self.decoder['candidate_value'].load_state_dict(value_static)

    def prepare_for_cls_encoding(self, input_ids, token_type_ids, attention_mask):
        # (all_candidate_num, max_seq_len)
        candidate_seq_len = input_ids.shape[-1]

        input_ids = input_ids.reshape(-1, candidate_seq_len)
        token_type_ids = token_type_ids.reshape(-1, candidate_seq_len)
        attention_mask = attention_mask.reshape(-1, candidate_seq_len)

        # remove cls
        input_ids = input_ids[:, 1:]
        token_type_ids = token_type_ids[:, 1:]
        attention_mask = attention_mask[:, 1:]

        # remove padding
        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        # remove last token in order to prepend context_cls
        if candidate_seq_len > self.config.text_max_len + 1 - self.config.context_num:
            remove_len = candidate_seq_len - (self.config.text_max_len + 1 - self.config.context_num)
            input_ids = input_ids[:, :-remove_len]
            token_type_ids = token_type_ids[:, :-remove_len]
            attention_mask = attention_mask[:, :-remove_len]

        # prepend context_cls
        input_embeds = self.embeddings(input_ids)
        expanded_context_cls = self.context_cls.unsqueeze(0).repeat(input_ids.shape[0], 1, 1)

        # get embedding
        input_embeds = torch.cat((expanded_context_cls, input_embeds), dim=1)

        # process mask and type id
        temp_zeros = torch.zeros((token_type_ids.shape[0], self.config.context_num),
                                 device=input_embeds.device, dtype=torch.int)
        temp_ones = torch.ones((attention_mask.shape[0], self.config.context_num),
                               device=input_embeds.device, dtype=torch.int)
        token_type_ids = torch.cat((temp_zeros, token_type_ids), dim=1)
        attention_mask = torch.cat((temp_ones, attention_mask), dim=1)

        return input_embeds, token_type_ids, attention_mask

    def prepare_candidates(self, input_ids, token_type_ids, attention_mask):
        """ Pre-compute representations. Return shape: (-1, context_num, context_dim) """

        input_embeds, token_type_ids, attention_mask =  self.prepare_for_cls_encoding(input_ids, token_type_ids, attention_mask)

        out = self.bert_model(inputs_embeds=input_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask,
                              enrich_candidate_by_question=[False]*self.this_bert_config.num_hidden_layers)
        last_hidden_state = out['last_hidden_state']

        candidate_embeddings = last_hidden_state[:, :self.config.context_num, :]
        return candidate_embeddings

    def do_queries_match(self, input_ids, token_type_ids, attention_mask, candidate_context_embeddings, **kwargs):
        """ use pre-compute candidate to get scores."""
        # used to control ablation
        no_aggregator = kwargs.get('no_aggregator', False)
        no_enricher = kwargs.get('no_enricher', False)

        # (query_num (batch_size), candidate_num, context_num, embedding_len)
        candidate_num = candidate_context_embeddings.shape[1]
        # reshape to (query_num, candidate_context_num, embedding_len)
        this_candidate_context_embeddings = candidate_context_embeddings.reshape(candidate_context_embeddings.shape[0],
                                                                                 -1,
                                                                                 candidate_context_embeddings.shape[-1])

        lstm_initial_state = torch.zeros(this_candidate_context_embeddings.shape[0], candidate_num,
                                         this_candidate_context_embeddings.shape[-1],
                                         device=this_candidate_context_embeddings.device)

        # (query_num, candidate_context_num + candidate_num, dim)
        this_candidate_input_embeddings = torch.cat((this_candidate_context_embeddings, lstm_initial_state), dim=1)

        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        a_out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                enrich_candidate_by_question=self.enrich_flag, candidate_embeddings=this_candidate_input_embeddings,
                                decoder=self.decoder, candidate_num=candidate_num)

        a_last_hidden_state, decoder_output = a_out['last_hidden_state'], a_out['decoder_output']

        # get final representations
        # control ablation
        if no_enricher:
            candidate_embeddings = self.decoder['candidate_composition_layer'](
                candidate_context_embeddings).squeeze(-2)
        else:
            # (query_num, candidate_num, context_num, dim)
            new_candidate_context_embeddings = decoder_output[:, :-candidate_num, :].reshape(decoder_output.shape[0],
                                                                                             candidate_num,
                                                                                             self.config.context_num,
                                                                                             decoder_output.shape[-1])

            # (query_num, candidate_num, dim)
            candidate_embeddings = self.decoder['candidate_composition_layer'](new_candidate_context_embeddings).squeeze(-2)

        if no_aggregator:
            # (query_num, 1, dim)
            query_embeddings = a_last_hidden_state[:, 0, :].unsqueeze(1)
            dot_product = torch.matmul(query_embeddings, candidate_embeddings.permute(0, 2, 1)).squeeze(-2)
        else:
            # (query_num, candidate_num, dim)
            query_embeddings = decoder_output[:, -candidate_num:, :]
            # (query_num, candidate_num)
            dot_product = torch.mul(query_embeddings, candidate_embeddings).sum(-1)

        return dot_product

    # 如果不把a传进q，就会退化成最普通的QAModel，已经被验证过了
    # b_text can be pre-computed
    def forward(self, a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, train_flag, **kwargs):
        """
        Forward can be used to route the function call to other functions in this Class.
        if train_flag==True, a_input_ids.shape and b_input_ids.shape are 2 dims, like (batch size, seq_len),
        otherwise, b_input_ids should look like (batch size, candidate num, seq len).
        """
        # Route to other functions,
        # since data parallel can only be triggered by forward
        do_queries_match = kwargs.get('do_queries_match', False)
        if do_queries_match:
            return self.do_queries_match(input_ids=a_input_ids,
                                         token_type_ids=a_token_type_ids,
                                         attention_mask=a_attention_mask,
                                         candidate_context_embeddings=kwargs.get('candidate_context_embeddings'),
                                         no_aggregator=kwargs.get('no_aggregator'),
                                         no_enricher=kwargs.get('no_enricher'),
                                         train_flag=train_flag)

        prepare_candidates = kwargs.get('prepare_candidates', False)
        if prepare_candidates:
            return self.prepare_candidates(input_ids=a_input_ids,
                                           token_type_ids=a_token_type_ids,
                                           attention_mask=a_attention_mask)

        # common training ---------------------------------------------------------
        return_dot_product = kwargs.get('return_dot_product', False)

        # must be clarified for training
        no_aggregator = kwargs.get('no_aggregator')
        no_enricher = kwargs.get('no_enricher')

        # (all_candidate_num, context_num, dim)
        b_embeddings = self.prepare_candidates(input_ids=b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_attention_mask)

        # convert to (query_num, candidate_num, context_num, dim)
        if not train_flag:
            b_embeddings = b_embeddings.reshape(a_input_ids.shape[0], -1, self.config.context_num, b_embeddings.shape[-1])
        else:
            # need broadcast
            b_embeddings = b_embeddings.reshape(-1, self.config.context_num, b_embeddings.shape[-1]).unsqueeze(0).\
                expand(a_input_ids.shape[0], -1, -1, -1)

        dot_product = self.do_queries_match(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                            attention_mask=a_attention_mask, candidate_context_embeddings=b_embeddings,
                                            no_aggregator=no_aggregator, no_enricher=no_enricher)

        if train_flag:
            if return_dot_product:
                return dot_product
            mask = torch.eye(a_input_ids.size(0)).to(a_input_ids.device)
            loss = F.log_softmax(dot_product, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()
            return loss
        else:
            return dot_product


# used for 1-n match task, disentangle two components
class DisenMatchParallelEncoder(nn.Module):
    def __init__(self, config: ParallelEncoderConfig):

        super(DisenMatchParallelEncoder, self).__init__()

        self.config = config

        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        self.this_bert_config = BertConfig.from_pretrained(config.pretrained_bert_path)

        # contain the parameters of encoder
        self.bert_model = MyBertModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)

        # used to compressing candidate to generate context vectors
        self.composition_layer = ContextLayer(self.this_bert_config.hidden_size, config.context_num)

        # info
        self.num_attention_heads = self.this_bert_config.num_attention_heads
        self.attention_head_size = int(self.this_bert_config.hidden_size / self.this_bert_config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # create models for decoder
        self.decoder = nn.ModuleDict({
            # used by candidate context embeddings to enrich themselves
            'candidate_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            'candidate_key': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                            range(self.this_bert_config.num_hidden_layers)]),
            'candidate_value': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            # used to project representations into same space
            'layer_chunks': nn.ModuleList(
                [DecoderLayerChunk(self.this_bert_config) for _ in range(self.this_bert_config.num_hidden_layers)]),
            # used to compress candidate context embeddings into a vector
            'candidate_composition_layer': SelectLastLayer(),
            # used to generate query to compress Text_A thus get its representation
            'compress_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                             range(self.this_bert_config.num_hidden_layers)]),
            # used to update hidden states with self-att output as input
            'LSTM': MyLSTMBlock(self.this_bert_config.hidden_size),
        })

        self.mean_layer = MeanLayer()

        # control which layer use decoder
        self.enrich_flag = [False] * self.this_bert_config.num_hidden_layers
        # self.enrich_flag[0] = True
        self.enrich_flag[-1] = True
        print("*" * 50)
        print(f"Enrich Layer: {self.enrich_flag}")
        print("*" * 50)

        self.init_decoder_params()

        # remove no use component
        for index, flag in enumerate(self.enrich_flag):
            if not flag:
                for key in ['layer_chunks', 'candidate_query', 'compress_query', 'candidate_key', 'candidate_value']:
                    self.decoder[key][index] = None

        print(f"Context Num: {self.config.context_num}, Note that 1 context will be used as hint!!!")
        print("*" * 50)

    # use parameters (layer chunk, q, q_1, k, v) of a pre-trained encoder to initialize the decoder
    def init_decoder_params(self):
        #  read parameters for layer chunks from encoder
        for index in range(self.this_bert_config.num_hidden_layers):
            this_static = self.decoder['layer_chunks'][index].state_dict()
            pretrained_static = self.bert_model.encoder.layer[index].state_dict()

            updating_layer_static = {}
            for k in this_static.keys():
                new_k = k.replace('attention.', 'attention.output.', 1)
                updating_layer_static[k] = pretrained_static[new_k]

            this_static.update(updating_layer_static)
            self.decoder['layer_chunks'][index].load_state_dict(this_static)

        #  read parameters for query from encoder
        pretrained_static = self.bert_model.state_dict()
        query_static = self.decoder['candidate_query'].state_dict()

        updating_query_static = {}
        for k in query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        query_static.update(updating_query_static)
        self.decoder['candidate_query'].load_state_dict(query_static)

        #  read parameters for compress_query from encoder
        compress_query_static = self.decoder['compress_query'].state_dict()

        updating_query_static = {}
        for k in compress_query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        compress_query_static.update(updating_query_static)
        self.decoder['compress_query'].load_state_dict(compress_query_static)

        #  read parameters for key from encoder
        key_static = self.decoder['candidate_key'].state_dict()

        updating_key_static = {}
        for k in key_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.key.' + k.split(".")[1]
            updating_key_static[k] = pretrained_static[full_key]

        key_static.update(updating_key_static)
        self.decoder['candidate_key'].load_state_dict(key_static)

        #  read parameters for value from encoder
        value_static = self.decoder['candidate_value'].state_dict()

        updating_value_static = {}
        for k in value_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.value.' + k.split(".")[1]
            updating_value_static[k] = pretrained_static[full_key]

        value_static.update(updating_value_static)
        self.decoder['candidate_value'].load_state_dict(value_static)

    def prepare_candidates(self, input_ids, token_type_ids, attention_mask):
        """ Pre-compute representations. Return shape: (-1, context_num, context_dim) """

        # reshape and encode candidates, (all_candidate_num, dim)
        candidate_seq_len = input_ids.shape[-1]
        input_ids = input_ids.reshape(-1, candidate_seq_len)
        token_type_ids = token_type_ids.reshape(-1, candidate_seq_len)
        attention_mask = attention_mask.reshape(-1, candidate_seq_len)

        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                              enrich_candidate_by_question=[False]*self.this_bert_config.num_hidden_layers)
        last_hidden_state = out['last_hidden_state']

        # get (all_candidate_num, context_num, dim)
        candidate_embeddings = self.composition_layer(last_hidden_state, attention_mask=attention_mask)

        return candidate_embeddings

    def do_queries_match(self, input_ids, token_type_ids, attention_mask, candidate_context_embeddings, **kwargs):
        """ use pre-compute candidate to get scores."""
        # used to control ablation
        no_aggregator = kwargs.get('no_aggregator', False)
        no_enricher = kwargs.get('no_enricher', False)

        # (query_num (batch_size), candidate_num, context_num, embedding_len)
        candidate_num = candidate_context_embeddings.shape[1]
        # reshape to (query_num, candidate_context_num, embedding_len)
        this_candidate_context_embeddings = candidate_context_embeddings.reshape(candidate_context_embeddings.shape[0],
                                                                                 -1,
                                                                                 candidate_context_embeddings.shape[-1])

        lstm_initial_state = torch.zeros(this_candidate_context_embeddings.shape[0], candidate_num, this_candidate_context_embeddings.shape[-1],
                                         device=this_candidate_context_embeddings.device)

        # (query_num, candidate_context_num + candidate_num, dim)
        this_candidate_input_embeddings = torch.cat((this_candidate_context_embeddings, lstm_initial_state), dim=1)

        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        a_out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                enrich_candidate_by_question=self.enrich_flag, candidate_embeddings=this_candidate_input_embeddings,
                                decoder=self.decoder, candidate_num=candidate_num)

        a_last_hidden_state, decoder_output = a_out['last_hidden_state'], a_out['decoder_output']

        # get final representations
        # control ablation
        if no_enricher:
            # if only one context vector, it is used for both aggregating and enriching.
            # if there are more than one vector, we use the last to aggregate and use the other to enrich
            if self.config.context_num == 1:
                candidate_embeddings = candidate_context_embeddings.squeeze(-2)
            else:
                candidate_embeddings = self.mean_layer(candidate_context_embeddings[:, :, :-1, :]).squeeze(-2)
        else:
            # (query_num, candidate_num, context_num, dim)
            new_candidate_context_embeddings = decoder_output[:, :-candidate_num, :].reshape(
                decoder_output.shape[0],
                candidate_num,
                self.config.context_num,
                decoder_output.shape[-1])

            # if only one context vector, it is used for both aggregating and enriching.
            # if there are more than one vector, we use the last to aggregate and use the other to enrich
            if self.config.context_num == 1:
                # (query_num, candidate_num, dim)
                candidate_embeddings = new_candidate_context_embeddings.squeeze(-2)
            else:
                candidate_embeddings = self.mean_layer(new_candidate_context_embeddings[:, :, :-1, :]).squeeze(-2)

        if no_aggregator:
            # (query_num, 1, dim)
            query_embeddings = a_last_hidden_state[:, 0, :].unsqueeze(1)
            dot_product = torch.matmul(query_embeddings, candidate_embeddings.permute(0, 2, 1)).squeeze(-2)
        else:
            # (query_num, candidate_num, dim)
            query_embeddings = decoder_output[:, -candidate_num:, :]
            # (query_num, candidate_num)
            dot_product = torch.mul(query_embeddings, candidate_embeddings).sum(-1)

        return dot_product

    # 如果不把a传进q，就会退化成最普通的QAModel，已经被验证过了
    # b_text can be pre-computed
    def forward(self, a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, train_flag, **kwargs):
        """
        Forward can be used to route the function call to other functions in this Class.
        if train_flag==True, a_input_ids.shape and b_input_ids.shape are 2 dims, like (batch size, seq_len),
        otherwise, b_input_ids should look like (batch size, candidate num, seq len).
        """
        # Route to other functions,
        # since data parallel can only be triggered by forward
        do_queries_match = kwargs.get('do_queries_match', False)
        if do_queries_match:
            return self.do_queries_match(input_ids=a_input_ids,
                                         token_type_ids=a_token_type_ids,
                                         attention_mask=a_attention_mask,
                                         candidate_context_embeddings=kwargs.get('candidate_context_embeddings'),
                                         no_aggregator=kwargs.get('no_aggregator'),
                                         no_enricher=kwargs.get('no_enricher'),
                                         train_flag=train_flag)

        prepare_candidates = kwargs.get('prepare_candidates', False)
        if prepare_candidates:
            return self.prepare_candidates(input_ids=a_input_ids,
                                           token_type_ids=a_token_type_ids,
                                           attention_mask=a_attention_mask)

        # common training ---------------------------------------------------------
        return_dot_product = kwargs.get('return_dot_product', False)

        # must be clarified for training
        no_aggregator = kwargs.get('no_aggregator')
        no_enricher = kwargs.get('no_enricher')

        # (all_candidate_num, context_num, dim)
        b_embeddings = self.prepare_candidates(input_ids=b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_attention_mask)

        # convert to (query_num, candidate_num, context_num, dim)
        if not train_flag:
            b_embeddings = b_embeddings.reshape(a_input_ids.shape[0], -1, self.config.context_num, b_embeddings.shape[-1])
        else:
            # need broadcast
            b_embeddings = b_embeddings.reshape(-1, self.config.context_num, b_embeddings.shape[-1]).unsqueeze(0).\
                expand(a_input_ids.shape[0], -1, -1, -1)

        dot_product = self.do_queries_match(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                            attention_mask=a_attention_mask, candidate_context_embeddings=b_embeddings,
                                            no_aggregator=no_aggregator, no_enricher=no_enricher)

        if train_flag:
            if return_dot_product:
                return dot_product
            mask = torch.eye(a_input_ids.size(0)).to(a_input_ids.device)
            loss = F.log_softmax(dot_product, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()
            return loss
        else:
            return dot_product


# used for 1-n match task
class DisenCLSMatchParallelEncoder(nn.Module):
    def __init__(self, config: ParallelEncoderConfig):

        super(DisenCLSMatchParallelEncoder, self).__init__()

        self.config = config

        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        self.this_bert_config = BertConfig.from_pretrained(config.pretrained_bert_path)

        # contain the parameters of encoder
        self.bert_model = MyBertModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)
        self.embeddings = self.bert_model.get_input_embeddings()

        # info
        self.num_attention_heads = self.this_bert_config.num_attention_heads
        self.attention_head_size = int(self.this_bert_config.hidden_size / self.this_bert_config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # contain cls for extract context vectors
        self.context_cls = nn.Parameter(torch.nn.init.xavier_normal_(torch.randn((self.config.context_num, self.all_head_size))))

        # create models for decoder
        self.decoder = nn.ModuleDict({
            # used by candidate context embeddings to enrich themselves
            'candidate_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            'candidate_key': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                            range(self.this_bert_config.num_hidden_layers)]),
            'candidate_value': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            # used to project representations into same space
            'layer_chunks': nn.ModuleList(
                [DecoderLayerChunk(self.this_bert_config) for _ in range(self.this_bert_config.num_hidden_layers)]),
            # used to compress candidate context embeddings into a vector
            'candidate_composition_layer': SelectLastLayer(),
            # used to generate query to compress Text_A thus get its representation
            'compress_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                             range(self.this_bert_config.num_hidden_layers)]),
            # used to update hidden states with self-att output as input
            'LSTM': MyLSTMBlock(self.this_bert_config.hidden_size),
        })

        self.mean_layer = MeanLayer()

        # self.classifier = SimpleClassifier(self.all_head_size)

        # control which layer use decoder
        self.enrich_flag = [False] * self.this_bert_config.num_hidden_layers
        
        if self.config.used_layers is None:
            self.enrich_flag[-1] = True
        elif self.config.used_layers == "pass":
            pass
        elif ',' in self.config.used_layers:
            layer_indices = [eval(temp_i) for temp_i in self.config.used_layers.split(",")]
            for l_i in layer_indices:
                self.enrich_flag[l_i] = True
        elif '-' in self.config.used_layers:
            layer_indices = [eval(temp_i) for temp_i in self.config.used_layers.split("-")]
            
            if len(layer_indices) != 2:
                raise Exception(f"You should offer used_layers as x-x, your given arg is {self.config.used_layers}.")
            
            for l_i in range(layer_indices[0], layer_indices[1] + 1):
                self.enrich_flag[l_i] = True
        else:
            self.enrich_flag[eval(self.config.used_layers)] = True

        # if last interaction layer is not the last Transformer layer, we can exit early
        self.last_layer_index = -1
        for temp_i, flag in enumerate(self.enrich_flag):
            if flag:
                self.last_layer_index = temp_i
                
        print("*" * 50)
        print(f"Enrich Layer: {self.enrich_flag}")
        print("*" * 50)

        self.init_decoder_params()

        # remove no use component
        for index, flag in enumerate(self.enrich_flag):
            if not flag:
                for key in ['layer_chunks', 'candidate_query', 'compress_query', 'candidate_key', 'candidate_value']:
                    self.decoder[key][index] = None

        print(f"Context Num: {self.config.context_num}, Note that 1 context will be used as hint!!!")
        print("*" * 50)

    # use parameters (layer chunk, q, q_1, k, v) of a pre-trained encoder to initialize the decoder
    def init_decoder_params(self):
        #  read parameters for layer chunks from encoder
        for index in range(self.this_bert_config.num_hidden_layers):
            this_static = self.decoder['layer_chunks'][index].state_dict()
            pretrained_static = self.bert_model.encoder.layer[index].state_dict()

            updating_layer_static = {}
            for k in this_static.keys():
                new_k = k.replace('attention.', 'attention.output.', 1)
                updating_layer_static[k] = pretrained_static[new_k]

            this_static.update(updating_layer_static)
            self.decoder['layer_chunks'][index].load_state_dict(this_static)

        #  read parameters for query from encoder
        pretrained_static = self.bert_model.state_dict()
        query_static = self.decoder['candidate_query'].state_dict()

        updating_query_static = {}
        for k in query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        query_static.update(updating_query_static)
        self.decoder['candidate_query'].load_state_dict(query_static)

        #  read parameters for compress_query from encoder
        compress_query_static = self.decoder['compress_query'].state_dict()

        updating_query_static = {}
        for k in compress_query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        compress_query_static.update(updating_query_static)
        self.decoder['compress_query'].load_state_dict(compress_query_static)

        #  read parameters for key from encoder
        key_static = self.decoder['candidate_key'].state_dict()

        updating_key_static = {}
        for k in key_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.key.' + k.split(".")[1]
            updating_key_static[k] = pretrained_static[full_key]

        key_static.update(updating_key_static)
        self.decoder['candidate_key'].load_state_dict(key_static)

        #  read parameters for value from encoder
        value_static = self.decoder['candidate_value'].state_dict()

        updating_value_static = {}
        for k in value_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.value.' + k.split(".")[1]
            updating_value_static[k] = pretrained_static[full_key]

        value_static.update(updating_value_static)
        self.decoder['candidate_value'].load_state_dict(value_static)

    def prepare_for_cls_encoding(self, input_ids, token_type_ids, attention_mask):
        # (all_candidate_num, max_seq_len)
        candidate_seq_len = input_ids.shape[-1]

        input_ids = input_ids.reshape(-1, candidate_seq_len)
        token_type_ids = token_type_ids.reshape(-1, candidate_seq_len)
        attention_mask = attention_mask.reshape(-1, candidate_seq_len)

        # remove cls
        input_ids = input_ids[:, 1:]
        token_type_ids = token_type_ids[:, 1:]
        attention_mask = attention_mask[:, 1:]

        # remove padding
        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        # remove last token in order to prepend context_cls
        if candidate_seq_len > self.config.text_max_len + 1 - self.config.context_num:
            remove_len = candidate_seq_len - (self.config.text_max_len + 1 - self.config.context_num)
            input_ids = input_ids[:, :-remove_len]
            token_type_ids = token_type_ids[:, :-remove_len]
            attention_mask = attention_mask[:, :-remove_len]

        # prepend context_cls
        input_embeds = self.embeddings(input_ids)
        expanded_context_cls = self.context_cls.unsqueeze(0).repeat(input_ids.shape[0], 1, 1)

        # get embedding
        input_embeds = torch.cat((expanded_context_cls, input_embeds), dim=1)

        # process mask and type id
        temp_zeros = torch.zeros((token_type_ids.shape[0], self.config.context_num),
                                 device=input_embeds.device, dtype=torch.int)
        temp_ones = torch.ones((attention_mask.shape[0], self.config.context_num),
                               device=input_embeds.device, dtype=torch.int)
        token_type_ids = torch.cat((temp_zeros, token_type_ids), dim=1)
        attention_mask = torch.cat((temp_ones, attention_mask), dim=1)

        return input_embeds, token_type_ids, attention_mask

    def prepare_candidates(self, input_ids, token_type_ids, attention_mask):
        """ Pre-compute representations. Return shape: (-1, context_num, context_dim) """

        input_embeds, token_type_ids, attention_mask =  self.prepare_for_cls_encoding(input_ids, token_type_ids, attention_mask)

        out = self.bert_model(inputs_embeds=input_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask,
                              enrich_candidate_by_question=[False]*self.this_bert_config.num_hidden_layers)
        last_hidden_state = out['last_hidden_state']

        candidate_embeddings = last_hidden_state[:, :self.config.context_num, :]
        return candidate_embeddings

    def do_queries_match(self, input_ids, token_type_ids, attention_mask, candidate_context_embeddings, **kwargs):
        """ use pre-compute candidate to get scores."""
        # used to control ablation
        no_aggregator = kwargs.get('no_aggregator', False)
        no_enricher = kwargs.get('no_enricher', False)

        # (query_num (batch_size), candidate_num, context_num, embedding_len)
        candidate_num = candidate_context_embeddings.shape[1]
        # reshape to (query_num, candidate_context_num, embedding_len)
        this_candidate_context_embeddings = candidate_context_embeddings.reshape(candidate_context_embeddings.shape[0],
                                                                                 -1,
                                                                                 candidate_context_embeddings.shape[-1])

        lstm_initial_state = torch.zeros(this_candidate_context_embeddings.shape[0], candidate_num,
                                         this_candidate_context_embeddings.shape[-1],
                                         device=this_candidate_context_embeddings.device)

        # (query_num, candidate_context_num + candidate_num, dim)
        this_candidate_input_embeddings = torch.cat((this_candidate_context_embeddings, lstm_initial_state), dim=1)

        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        a_out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                enrich_candidate_by_question=self.enrich_flag, candidate_embeddings=this_candidate_input_embeddings,
                                decoder=self.decoder, candidate_num=candidate_num, last_layer_index=self.last_layer_index)

        a_last_hidden_state, decoder_output = a_out['last_hidden_state'], a_out['decoder_output']

        # get final representations
        # control ablation
        if no_enricher:
            # if only one context vector, it is used for both aggregating and enriching.
            # if there are more than one vector, we use the last to aggregate and use the other to enrich
            if self.config.context_num == 1:
                candidate_embeddings = candidate_context_embeddings.squeeze(-2)
            else:
                candidate_embeddings = self.mean_layer(candidate_context_embeddings[:, :, :-1, :]).squeeze(-2)
        else:
            # (query_num, candidate_num, context_num, dim)
            new_candidate_context_embeddings = decoder_output[:, :-candidate_num, :].reshape(decoder_output.shape[0],
                                                                                             candidate_num,
                                                                                             self.config.context_num,
                                                                                             decoder_output.shape[-1])

            # if only one context vector, it is used for both aggregating and enriching.
            # if there are more than one vector, we use the last to aggregate and use the other to enrich
            if self.config.context_num == 1:
                # (query_num, candidate_num, dim)
                candidate_embeddings = new_candidate_context_embeddings.squeeze(-2)
            else:
                candidate_embeddings = self.mean_layer(new_candidate_context_embeddings[:, :, :-1, :]).squeeze(-2)

        # dot_product = self.classifier(candidate_embeddings).squeeze(-1)
        # return dot_product

        if no_aggregator:
            # (query_num, 1, dim)
            query_embeddings = a_last_hidden_state[:, 0, :].unsqueeze(1)
            dot_product = torch.matmul(query_embeddings, candidate_embeddings.permute(0, 2, 1)).squeeze(-2)
        else:
            # (query_num, candidate_num, dim)
            query_embeddings = decoder_output[:, -candidate_num:, :]
            # (query_num, candidate_num)
            dot_product = torch.mul(query_embeddings, candidate_embeddings).sum(-1)

        return dot_product

    # 如果不把a传进q，就会退化成最普通的QAModel，已经被验证过了
    # b_text can be pre-computed
    def forward(self, a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, train_flag, **kwargs):
        """
        Forward can be used to route the function call to other functions in this Class.
        if train_flag==True, a_input_ids.shape and b_input_ids.shape are 2 dims, like (batch size, seq_len),
        otherwise, b_input_ids should look like (batch size, candidate num, seq len).
        """
        # Route to other functions,
        # since data parallel can only be triggered by forward
        do_queries_match = kwargs.get('do_queries_match', False)
        if do_queries_match:
            return self.do_queries_match(input_ids=a_input_ids,
                                         token_type_ids=a_token_type_ids,
                                         attention_mask=a_attention_mask,
                                         candidate_context_embeddings=kwargs.get('candidate_context_embeddings'),
                                         no_aggregator=kwargs.get('no_aggregator'),
                                         no_enricher=kwargs.get('no_enricher'),
                                         train_flag=train_flag)

        prepare_candidates = kwargs.get('prepare_candidates', False)
        if prepare_candidates:
            return self.prepare_candidates(input_ids=a_input_ids,
                                           token_type_ids=a_token_type_ids,
                                           attention_mask=a_attention_mask)

        # common training ---------------------------------------------------------
        return_dot_product = kwargs.get('return_dot_product', False)

        # must be clarified for training
        no_aggregator = kwargs.get('no_aggregator')
        no_enricher = kwargs.get('no_enricher')

        # (all_candidate_num, context_num, dim)
        b_embeddings = self.prepare_candidates(input_ids=b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_attention_mask)

        # convert to (query_num, candidate_num, context_num, dim)
        if not train_flag:
            b_embeddings = b_embeddings.reshape(a_input_ids.shape[0], -1, self.config.context_num, b_embeddings.shape[-1])
        else:
            # need broadcast
            b_embeddings = b_embeddings.reshape(-1, self.config.context_num, b_embeddings.shape[-1]).unsqueeze(0).\
                expand(a_input_ids.shape[0], -1, -1, -1)

        dot_product = self.do_queries_match(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                            attention_mask=a_attention_mask, candidate_context_embeddings=b_embeddings,
                                            no_aggregator=no_aggregator, no_enricher=no_enricher)

        if train_flag:
            if return_dot_product:
                return dot_product
            mask = torch.eye(a_input_ids.size(0)).to(a_input_ids.device)
            loss = F.log_softmax(dot_product, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()
            return loss
        else:
            return dot_product


# --------------------------------------
# fen ge xian
# --------------------------------------
# A gate, a substitution of resnet
class Classifier(nn.Module):
    def __init__(self, input_len, keep_prob=0.9, num_labels=4):
        super(Classifier, self).__init__()

        self.linear1 = torch.nn.Linear(input_len, input_len * 2, bias=True)
        self.linear2 = torch.nn.Linear(input_len * 2, num_labels, bias=True)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=1 - keep_prob)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, a_embedding):
        x = a_embedding
        # x = torch.cat((q_embedding, a_embedding), dim=-1)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)

        return x


class SimpleClassifier(nn.Module):
    def __init__(self, input_len):
        super(SimpleClassifier, self).__init__()

        self.linear1 = torch.nn.Linear(input_len, 1, bias=False)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight)

    def forward(self, a_embedding):
        x = a_embedding

        x = self.linear1(x)

        return x


class MyLSTMBlock(nn.Module):
    def __init__(self, input_dim):
        super(MyLSTMBlock, self).__init__()
        self.weight_layer = nn.Linear(input_dim*3, input_dim)

        self.sigmoid = nn.Sigmoid()

    # shapes are supposed to be (batch size, input_dim)
    def forward(self, this_compressed_vector, last_compressed_vector, weight_hint):
        weight = self.weight_layer(torch.cat((weight_hint, this_compressed_vector, last_compressed_vector), dim=-1))
        weight = self.sigmoid(weight)

        new_compressed_vector = weight * this_compressed_vector + (1-weight)*last_compressed_vector

        return new_compressed_vector


class ContextLayer(nn.Module):
    def __init__(self, input_dim, context_num):
        super(ContextLayer, self).__init__()
        self.query = nn.Parameter(torch.randn(context_num, input_dim))

    def forward(self, context, attention_mask=None):
        """
        :param context: (..., sequence len, input_dim)
        :param attention_mask: (..., sequence len)
        :return: (..., context_num, input_dim)
        """

        context_representation = dot_attention(q=self.query, k=context, v=context, v_mask=attention_mask)

        return context_representation


class MeanLayer(nn.Module):
    def __init__(self):
        super(MeanLayer, self).__init__()

    @staticmethod
    def forward(embeddings, token_type_ids=None, attention_mask=None):
        """
        do average at dim -2 without remove this dimension
        :param embeddings: (..., sequence len, dim)
        :param token_type_ids: optional
        :param attention_mask: optional
        :return: (..., 1, dim)
        """

        representation = get_rep_by_avg(embeddings, token_type_ids, attention_mask)

        return representation


class SelectLastLayer(nn.Module):
    def __init__(self):
        super(SelectLastLayer, self).__init__()

    @staticmethod
    def forward(embeddings, token_type_ids=None, attention_mask=None):
        """
        do average at dim -2 without remove this dimension
        :param embeddings: (..., sequence len, dim)
        :param token_type_ids: optional
        :param attention_mask: optional
        :return: (..., 1, dim)
        """

        if embeddings.dim() == 3:
            representation = embeddings[:, -1:, :]
        elif embeddings.dim() == 4:
            representation = embeddings[:, :, -1:, :]
        else:
            raise Exception("representation dim should be 3 or 4!")

        return representation