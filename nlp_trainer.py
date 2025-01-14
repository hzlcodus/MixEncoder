# %%
import gc
import math
import time

import datasets
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional
import os
import torch.utils.data
from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import sys
from tqdm import tqdm, trange
import csv
import torch.nn.functional as F
import jsonlines
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel, convert_syncbn_model
    APEX_FLAG = True
    print("*"*50)
    print("Apex Available")
    print("*"*50)
except:
    APEX_FLAG = False

from nlp_model import QAClassifierModel, QAClassifierModelConfig, CrossBERT, CrossBERTConfig, ClassifyParallelEncoder, \
    ParallelEncoderConfig, PolyEncoder, PolyEncoderConfig, QAMatchModel, CMCModel, MatchParallelEncoder, ClassifyDeformer, \
    DeformerConfig, MatchDeformer, MatchCrossBERT, ColBERT, ColBERTConfig
from my_function import sum_average_tuple, raise_dataset_error, print_recall_precise, print_optimizer, \
    raise_test_error, tokenize_and_truncate_from_head, get_elapse_time, calculate_recall, calculate_mrr, load_qrels
from nlp_dataset import SingleInputDataset, DoubleInputDataset, DoubleInputLabelDataset, SingleInputLabelDataset, \
    MSMARCODoubleInputDataset, MSMARCODataset
from novel_model import CLSMatchParallelEncoder, CLSClassifyParallelEncoder, DisenMatchParallelEncoder, \
    DisenCLSMatchParallelEncoder, DisenCLSClassifyParallelEncoder, DisenClassifyParallelEncoder

root_path = "/data/cme/fast_match/"

class TrainWholeModel:
    def __init__(self, args, config=None):

        # hyper-parameter
        self.real_test_data_split_num = 50

        # 读取一些参数并存起来-------------------------------------------------------------------
        self.__read_args_for_train(args)
        self.args = args

        # 设置gpu-------------------------------------------------------------------
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        # os.environ["TOKENIZERS_PARALLELISM"] = "false"
        #os.environ["CUDA_VISIBLE_DEVICES"] = args.nvidia_number

        # for data_parallel-------------------------------------------------------------------
        nvidia_number = len(args.nvidia_number.split(","))
        self.device_ids = [i for i in range(nvidia_number)]

        # self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # if not torch.cuda.is_available():
        #     raise Exception("No cuda available!")

        local_rank = 0
        if self.data_distribute:
            local_rank = self.local_rank
            torch.cuda.set_device(local_rank)
            torch.distributed.init_process_group(
                'nccl',
                init_method='env://'
            )
        else:
            torch.cuda.set_device(local_rank)

        if self.args.use_cpu:
            self.device = torch.device(f'cpu')
            print("Using cpu!!!!!")
        else:
            print(f"local rank: {local_rank}")
            self.device = torch.device(f'cuda:{local_rank}')

        # 读取tokenizer-------------------------------------------------------------------
        tokenizer_path = args.pretrained_bert_path.replace("/", "_")
        tokenizer_path = tokenizer_path.replace("\\", "_")

        # read from disk or save to disk
        if os.path.exists("../tokenizer/" + tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained("../tokenizer/" + tokenizer_path)
        else:
            print("first time use this tokenizer, downloading...")
            self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_bert_path)
            tokenizer_config = AutoConfig.from_pretrained(args.pretrained_bert_path)

            self.tokenizer.save_pretrained("../tokenizer/" + tokenizer_path)
            tokenizer_config.save_pretrained("../tokenizer/" + tokenizer_path)

        # 获得模型配置-------------------------------------------------------------------
        if config is None:
            self.config = self.__read_args_for_config(args)
        else:
            self.config = config

        # avoid warning
        self.model = None
        self.teacher_model = None

    def train(self, train_two_stage_flag, only_final=False):
        # best model save path
        model_save_path = self.save_model_dict + "/" + self.model_save_prefix + self.model_class + "_" + \
                          self.dataset_name
        # last model save path
        last_model_save_path = self.last_model_dict + "/" + self.model_save_prefix + self.model_class + "_" + \
                               self.dataset_name

        # 用来判断是哪一阶段的训练
        final_stage_flag = not train_two_stage_flag

        # two stage training but begin with final stage
        if only_final:
            final_stage_flag = True

        previous_best_performance = 0.0

        while True:
            if train_two_stage_flag:
                if final_stage_flag:
                    print("~"*60 + " begin final stage train " + "~"*60)
                else:
                    print("~"*60 + " begin first stage train " + "~"*60)
            else:
                print("~"*60 + " begin one stage train " + "~"*60)

            # create model
            self.model = self.__create_model()

            # prepare for training
            self.__model_to_device()
            train_step_function = self.select_train_step_function()

            # get optimizer
            optimizer = self.__get_optimizer(final_stage_flag=final_stage_flag)

            # 设置早停变量
            if final_stage_flag:
                if self.dataset_name in ['msmarco', 'hard_msmarco']:
                    early_stop_threshold = 2
                else:
                    early_stop_threshold = 3
            else:
                early_stop_threshold = 1

            # get restore epoch
            restore_path = self.get_restore_path(final_stage_flag)

            if self.restore_flag:
                restore_epoch = torch.load(restore_path)['epoch']
            else:
                restore_epoch = 0

            # prepare dataset, all are tuples
            train_dataset, val_datasets, test_datasets = self.__get_datasets()

            # print matching dataset info
            if self.model_class in ["MatchParallelEncoder", "CLSMatchParallelEncoder", 'DisenMatchParallelEncoder',
                                    "CLSClassifyParallelEncoder", "ClassifyParallelEncoder", 'DisenCLSMatchParallelEncoder',
                                    'DisenCLSClassifyParallelEncoder', 'DisenClassifyParallelEncoder']:
                if self.dataset_name in ['msmarco', 'hard_msmarco']:
                    pass
                else:
                    self.print_dataset_info(train_dataset, test_datasets)
            print("*"*50)

            # avoid warning
            scheduler = None
            has_best_model = False

            for epoch in range(restore_epoch, self.num_train_epochs):
                torch.cuda.empty_cache()

                # whether this epoch get the best model
                this_epoch_best = False

                # 打印一下
                print("*" * 20 + f" {epoch + 1} " + "*" * 20)

                # get train dataloader
                train_dataloader = self.__get_dataloader(data=train_dataset, batch_size=self.train_batch_size,
                                                         train_flag=True)[0]

                if self.data_distribute:
                    train_dataloader.sampler.set_epoch(epoch)

                # 获取scheduler, 并设置好apex, 并行，然后读取模型
                if epoch == restore_epoch:
                    if self.data_distribute:
                        self.model = convert_syncbn_model(self.model)
                    if APEX_FLAG and not self.no_apex:
                        self.model, optimizer = amp.initialize(self.model, optimizer, opt_level="O1")
                    if self.data_distribute:
                        self.model = DistributedDataParallel(self.model, delay_allreduce=True)
                    if self.data_parallel:
                        print("Let's use", torch.cuda.device_count(), "GPUs!")
                        self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
                    # self.__model_parallel()

                    # 初始化好之后开始读取模型
                    # 如果要进行第二阶段训练，那需要先读取上一阶段最好的model, for normal two stage
                    if final_stage_flag and train_two_stage_flag:
                        self.model = self.load_model(self.model, model_save_path + "_middle")

                    # load model to restore training
                    if self.restore_flag:
                        self.model = self.load_model(self.model, restore_path)

                    # restore training settings
                    early_stop_count, restore_epoch, scheduler_last_epoch, previous_best_performance = \
                        self.restore_settings(optimizer, restore_path, previous_best_performance)

                    scheduler = self.get_scheduler(optimizer, scheduler_last_epoch, train_dataloader)

                # 训练之前先看初始情况,保存初始best performance
                if epoch == restore_epoch and not self.no_initial_test:
                    print("-" * 30 + "initial validation" + "-" * 30)

                    this_best_performance = self.do_val(val_datasets, previous_best_performance)

                    if this_best_performance > previous_best_performance:
                        previous_best_performance = this_best_performance
                    print("-" * 30 + "initial_test_end" + "-" * 30, end="\n\n")

                # 开始训练
                train_loss = 0.0
                # 计算训练集的acc
                shoot_num, hit_num, now_batch_num = 0, 0, 0

                self.model.train()

                # 开始训练----------------------------------------------------------------------------------------
                # 进度条
                bar = tqdm(train_dataloader, total=len(train_dataloader))

                # 开始训练
                for batch in bar:
                    step_train_returns = train_step_function(batch=batch,
                                                             optimizer=optimizer,
                                                             now_batch_num=now_batch_num,
                                                             scheduler=scheduler,
                                                             final_stage_flag=final_stage_flag)

                    # match bi training
                    if len(step_train_returns) == 1:
                        step_loss = step_train_returns[0]

                        # 更新一下信息
                        train_loss += step_loss.item()
                        now_batch_num += 1

                        bar.set_description(
                            "epoch {:>3d} loss {:.4f}".format(epoch + 1, train_loss / now_batch_num))
                    else:
                        step_loss, step_shoot_num, step_hit_num = step_train_returns

                        # 更新一下信息
                        shoot_num += step_shoot_num
                        hit_num += step_hit_num
                        train_loss += step_loss.item()
                        now_batch_num += 1

                        bar.set_description(
                            "epoch {:>3d} loss {:.4f} Acc(R@1) {:>10d}/{:>10d} = {:.4f}".format(epoch + 1,
                                                                                                train_loss / now_batch_num,
                                                                                                hit_num, shoot_num,
                                                                                                hit_num / shoot_num * 100))

                    # 是否评测一次
                    # 忽略，变更为一次epoch结束评测一次
                    # ......

                print("epoch {:>3d} loss {:.4f}".format(epoch + 1, train_loss / now_batch_num))

                # 准备存储模型
                postfix = ""
                if not final_stage_flag:
                    postfix = "_middle"

                # 先保存模型，免得评测代码出错 
                self.save_model(model_save_path=last_model_save_path + postfix, epoch=epoch, optimizer=optimizer,
                                scheduler=scheduler,
                                previous_best_performance=previous_best_performance, early_stop_count=early_stop_count)

                # 验证集上测试
                if (epoch+1) % self.evaluate_epoch == 0:
                    this_best_performance = self.do_val(val_datasets, previous_best_performance)

                    # 存储最优模型
                    if this_best_performance > previous_best_performance:
                        previous_best_performance = this_best_performance
                        this_epoch_best = True
                        has_best_model = True

                        self.save_model(model_save_path=model_save_path + postfix, epoch=epoch, optimizer=optimizer,
                                        scheduler=scheduler,
                                        previous_best_performance=previous_best_performance,
                                        early_stop_count=early_stop_count)

                        if self.dataset_name in ['mnli', 'qqp', 'boolq']:
                            self.do_test(test_datasets=test_datasets, postfix=postfix)

                    # 是否早停
                    if this_epoch_best:
                        early_stop_count = 0
                    else:
                        early_stop_count += 1

                        if early_stop_count == early_stop_threshold:
                            print("early stop!")
                            #break #jcy : disable early stop

                torch.cuda.empty_cache()
                gc.collect()
                sys.stdout.flush()

            # 在测试集上做最后的检验
            # 用来修改预测的文件名
            postfix = ""
            if not final_stage_flag:
                postfix = "_middle"

            # 用最好的模型
            if has_best_model:
                self.model = self.load_model(self.model, model_save_path + postfix)

            if self.dataset_name in ["msmarco", "hard_msmarco"]:
                self.do_val(val_datasets, previous_best_performance)
            else:
                self.do_test(test_datasets=test_datasets, postfix=postfix, previous_best_performance=previous_best_performance)

            # 如果只是第一阶段的训练，那么还要继续训练
            if not final_stage_flag:
                final_stage_flag = True
            else:
                break

    def select_train_step_function(self):
        train_step_function = None

        # add model
        if self.model_class == "CrossBERT":
            if self.dataset_name in ['mnli', 'qqp', 'boolq']:
                train_step_function = self.__classify_train_step_for_cross
            elif self.dataset_name in ['dstc7', 'ubuntu', 'yahooqa']:
                train_step_function = self.__match_train_step_for_cross
            else:
                raise_dataset_error()
        elif self.model_class in ['QAClassifierModel', 'ClassifyParallelEncoder', 'PolyEncoder', 'QAMatchModel', 'CMCModel',
                                  'MatchParallelEncoder', 'ClassifyDeformer', 'MatchDeformer', 'MatchCrossBERT',
                                  'CLSMatchParallelEncoder', 'CLSClassifyParallelEncoder', 'DisenMatchParallelEncoder',
                                  'DisenCLSMatchParallelEncoder', 'DisenCLSClassifyParallelEncoder',
                                  'DisenClassifyParallelEncoder', 'ColBERT']:
            if self.dataset_name in ['mnli', 'qqp', 'boolq']:
                train_step_function = self.__classify_train_step_for_qa_input
            elif self.dataset_name in ['hard_msmarco']:
                train_step_function = self.__match_train_step_for_qa_input # jcy : CMCMODEL, 그리고 다른 모델에도 통용 가능하도록
                print("train_step_function is match_train_step_for_qa_input")
                # train_step_function = self.__clean_graph_efficient_match_train_step_for_bi_hard_msmarco #MSMARCOdataset에서 a, b, c 인풋 들어왔을 때. forward도 안 부름
                # print("train_step_function is clean_graph_efficient_match_train_step_for_bi_hard_msmarcorain_step_for_qa_input")
            elif (self.dataset_name in ['yahooqa', 'msmarco']) and (not self.in_batch):
                train_step_function = self.__train_step_for_multi_candidates_input
            elif self.dataset_name in ['dstc7', 'ubuntu', 'msmarco']:
                # support advance training function
                if self.model_class in ['MatchParallelEncoder', 'PolyEncoder', 'MatchDeformer', 'QAMatchModel', 
                                        'CLSMatchParallelEncoder', 'DisenMatchParallelEncoder',
                                        'DisenCLSMatchParallelEncoder', 'ColBERT']:
                    if self.data_parallel and self.model_class in ['MatchParallelEncoder', 'CLSMatchParallelEncoder',
                                                                   'DisenMatchParallelEncoder', 'DisenCLSMatchParallelEncoder']:
                        train_step_function = self.__data_parallel_match_train_step_for_qa_input
                        print("train_step_function is data_parallel_match_train_step_for_qa_input")
                    elif self.clean_graph:
                        train_step_function = self.__clean_graph_efficient_match_train_step_for_qa_input
                        print("train_step_function is clean_graph_efficient_match_train_step_for_qa_input")
                    else:
                        #train_step_function = self.__efficient_match_train_step_for_qa_input
                        #print("train_step_function is efficient_match_train_step_for_qa_input")
                        train_step_function = self.__match_train_step_for_qa_input # CMCMODEL
                        print("train_step_function is match_train_step_for_qa_input")

                # common training
                else:
                    train_step_function = self.__match_train_step_for_qa_input # CMCMODEL
                    print("train_step_function is match_train_step_for_qa_input")
            else:
                raise_dataset_error()
        else:
            raise Exception("Train step have not supported this model class")

       
        return train_step_function

    def get_scheduler(self, optimizer, scheduler_last_epoch, train_dataloader):
        num_epoch = self.num_train_epochs
        # if self.dataset_name == 'msmarco':
        #     num_epoch = 1
        print("*"*50)
        print(f"Training epoch is {num_epoch}")
        print("*"*50)

        t_total = (
                          len(train_dataloader) // self.gradient_accumulation_steps) * num_epoch
        if self.dataset_name == 'msmarco':
            warmup_step = int((t_total * 0.1) / num_epoch)
        else:
            warmup_step = int(t_total * 0.1)

        remain_step = t_total - scheduler_last_epoch

        # if restore training, should pass last epoch, otherwise should not pass this argument
        if self.restore_flag:
            if self.dataset_name == 'msmarco':
                scheduler = get_constant_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=warmup_step,
                                                            last_epoch=scheduler_last_epoch)
            else:
                scheduler = get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=warmup_step,
                                                            num_training_steps=t_total,
                                                            last_epoch=scheduler_last_epoch)
            # avoid trying to restore again
            self.restore_flag = False

            # print lr
            print_optimizer(optimizer)
        else:
            if self.dataset_name == 'msmarco':
                scheduler = get_constant_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=warmup_step)
            else:
                scheduler = get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=warmup_step,
                                                            num_training_steps=t_total)

        print(f"Train {self.num_train_epochs} epochs, Block num {self.dataset_split_num}, "
              f"Accumulate num {self.gradient_accumulation_steps}, Total update {t_total}, Remain update {remain_step}\n")

        return scheduler

    def do_val(self, val_datasets, previous_best_performance, **kwargs):
        self.model.eval()

        # clean cuda memory
        torch.cuda.empty_cache()
        gc.collect()

        # add dataset
        if self.dataset_name in ['mnli', 'qqp', 'boolq']:
            now_best_performance = self.classify_do_val_body(val_datasets, previous_best_performance)
        elif self.dataset_name in ['dstc7', 'ubuntu', 'yahooqa', 'msmarco', 'hard_msmarco']:
            now_best_performance = self.match_val_test_body(this_datasets=val_datasets,
                                                            previous_best_performance=previous_best_performance
                                                            , do_test=False)
        else:
            raise Exception("do_val is not supported for this dataset_name!")

        self.model.train()
        return now_best_performance

    def print_dataset_info(self, train_dataset, test_datasets):
        for print_dataset in train_dataset:
            a_attention_mask = print_dataset.a_attention_mask
            all_token_num = a_attention_mask.sum()
            avg_len = all_token_num / a_attention_mask.shape[0]
            print(f"Train query num is {a_attention_mask.shape[0]}. Avg len of train query is {avg_len}.", end="")

            b_attention_mask = print_dataset.b_attention_mask
            all_token_num = b_attention_mask.sum()
            # add dataset
            if self.dataset_name in ['yahooqa']:
                avg_len = all_token_num / (b_attention_mask.shape[0] * b_attention_mask.shape[1])
                print(
                    f" Candidate each query is {b_attention_mask.shape[1]}. Avg len of test candidate is {avg_len}.")
            else:
                avg_len = all_token_num / b_attention_mask.shape[0]
                print(f" Avg len of train candidate is {avg_len}.")

        for print_dataset in test_datasets:
            # datasets for cross will raise exception here
            a_attention_mask = print_dataset.a_attention_mask
            all_token_num = a_attention_mask.sum()
            avg_len = all_token_num / a_attention_mask.shape[0]
            print(f"Test query num is {a_attention_mask.shape[0]}. Avg len of test query is {avg_len}.", end="")

            b_attention_mask = print_dataset.b_attention_mask
            all_token_num = b_attention_mask.sum()

            # like some classification task
            if b_attention_mask.dim() == 2:
                avg_len = all_token_num / b_attention_mask.shape[0]
                print(f" Avg len of test candidate is {avg_len}.")
            else:
                avg_len = all_token_num / (b_attention_mask.shape[0] * b_attention_mask.shape[1])
                print(f" Candidate each query is {b_attention_mask.shape[1]}. Avg len of test candidate is {avg_len}.")

    def do_test(self, test_datasets=None, model_save_path=None, postfix="", **kwargs):
        # clean cuda memory
        torch.cuda.empty_cache()
        gc.collect()

        previous_best_performance = kwargs.get('previous_best_performance', 0.0)
        do_val = kwargs.get('do_val', False)

        if self.dataset_name in ['mnli', 'qqp', 'boolq']:
            self.glue_test(test_datasets=test_datasets, model_save_path=model_save_path, postfix=postfix, do_val=do_val)
        elif self.dataset_name in ['dstc7', 'ubuntu', 'yahooqa', 'msmarco', 'hard_msmarco']:
            _ = self.match_val_test_body(this_datasets=test_datasets,
                                         previous_best_performance=previous_best_performance,
                                         do_test=not do_val, model_save_path=model_save_path)
        else:
            raise Exception("do_val is not supported for this dataset_name!")

        self.model.train()
        return None

    def match_bi_real_test(self, test_datasets: DoubleInputDataset = None, model_save_path=None, **kwargs):
        """

        :param test_datasets: contain queries: (query_num, query_seq_len), contain candidates: (query_num, val_candidate_num, candidate_seq_len)
        :param model_save_path: use to load model if model is not created
        :param kwargs: placeholder
        :return: None
        """
        print("*" * 40 + " Begin Real Testing " + "*" * 40)

        # prepare data
        if os.path.exists(root_path + "dataset/" + self.dataset_name + "_real"):
            dataset = torch.load(root_path + "dataset/" + self.dataset_name + "_real")

            candidate_input_ids = dataset['candidate_input_ids']
            candidate_attention_mask = dataset['candidate_attention_mask']
            candidate_token_type_ids = dataset['candidate_token_type_ids']
            query_input_ids = dataset['query_input_ids']
            query_attention_mask = dataset['query_attention_mask']
            query_token_type_ids = dataset['query_token_type_ids']

            start_index = 0
            end_index = 100
            # (query_num, candidate_num, seq_len)
            # dstc: (1000, 100, 327)
            candidate_input_ids = candidate_input_ids[start_index:end_index]
            candidate_attention_mask = candidate_attention_mask[start_index:end_index]
            candidate_token_type_ids = candidate_token_type_ids[start_index:end_index]

            query_input_ids = query_input_ids[start_index:end_index]
            query_attention_mask = query_attention_mask[start_index:end_index]
            query_token_type_ids = query_token_type_ids[start_index:end_index]
        else:
            if not test_datasets:
                if self.dataset_name in ['msmarco', 'hard_msmarco']:
                    _, this_datasets, _ = self.__get_datasets(get_train=False, get_test=False)
                else:
                    _, _, this_datasets = self.__get_datasets(get_train=False, get_val=False)
            else:
                this_datasets = test_datasets

            if len(this_datasets) > 1:
                raise Exception("Test Datasets is more than 1 for match task!")

            this_datasets = this_datasets[0]

            start_index = 0
            end_index = 800
            # (query_num, candidate_num, seq_len)
            # dstc: (1000, 100, 327)
            candidate_input_ids = this_datasets.b_input_ids[start_index:end_index]
            candidate_attention_mask = this_datasets.b_attention_mask[start_index:end_index]
            candidate_token_type_ids = this_datasets.b_token_type_ids[start_index:end_index]

            query_input_ids = this_datasets.a_input_ids[start_index:end_index]
            query_attention_mask = this_datasets.a_attention_mask[start_index:end_index]
            query_token_type_ids = this_datasets.a_token_type_ids[start_index:end_index]

            print(candidate_input_ids.shape)

            torch.save(
                {"candidate_input_ids": candidate_input_ids, "candidate_attention_mask": candidate_attention_mask,
                 "candidate_token_type_ids": candidate_token_type_ids,
                 "query_input_ids": query_input_ids, "query_attention_mask": query_attention_mask,
                 "query_token_type_ids": query_token_type_ids}, root_path + "dataset/" + self.dataset_name + "_real")

        query_num = candidate_input_ids.shape[0]
        candidate_num = candidate_input_ids.shape[1]

        print(f"Query shape {query_input_ids.shape}, Candidate shape {candidate_input_ids.shape}")

        # create model if necessary
        if self.model is None:
            self.model = self.__create_model()
            # self.model = self.load_model(self.model, model_save_path)
            self.model.to(self.device)

        self.model.eval()

        with torch.no_grad():
            # pre-compute candidates
            if os.path.exists("./output/" + self.dataset_name + "_" + self.model_save_prefix + self.model_class):
                whole_candidate_embeddings = torch.load("./output/" + self.dataset_name + "_" + self.model_save_prefix +
                                                        self.model_class)
            else:
                whole_candidate_embeddings = []

                query_step = 10
                if self.dataset_name in ['msmarco', 'hard_msmarco']:
                    query_step = 3

                now_query_count = 0
                pbar = tqdm(total=query_num)
                # encoding one by one
                while now_query_count < query_num:
                    # (query_step, candidate_num, seq_len)
                    this_input_ids = candidate_input_ids[now_query_count:(now_query_count+query_step)].to(self.device)
                    this_attention_mask = candidate_attention_mask[now_query_count:(now_query_count+query_step)].to(self.device)
                    this_token_type_ids = candidate_token_type_ids[now_query_count:(now_query_count+query_step)].to(self.device)

                    this_batch_size = this_input_ids.shape[0]

                    if self.model_class in ['MatchDeformer', 'ColBERT']:
                        # (batch_size*candidate_num, seq_len, dim)
                        this_input_ids = this_input_ids.reshape(this_batch_size*candidate_num, this_input_ids.shape[-1])
                        this_attention_mask = this_attention_mask.reshape(this_batch_size*candidate_num, this_input_ids.shape[-1])
                        this_token_type_ids = this_token_type_ids.reshape(this_batch_size*candidate_num, this_input_ids.shape[-1])

                        return_value = self.model.prepare_candidates(input_ids=this_input_ids,
                                                                            token_type_ids=this_token_type_ids,
                                                                            attention_mask=this_attention_mask,
                                                                            clean_flag=False)
                        if isinstance(return_value, tuple):
                            batch_embeddings = return_value[0]
                        else:
                            batch_embeddings = return_value
                    else:
                        # (batch_size*candidate_num, (context_num), dim)
                        batch_embeddings = self.model.prepare_candidates(input_ids=this_input_ids,
                                                                        token_type_ids=this_token_type_ids,
                                                                        attention_mask=this_attention_mask)

                    if self.model_class in ['MatchParallelEncoder', 'MatchDeformer',
                                            'CLSMatchParallelEncoder', 'DisenMatchParallelEncoder',
                                            'DisenCLSMatchParallelEncoder', 'ColBERT']:
                        batch_embeddings = batch_embeddings.reshape(this_batch_size, candidate_num,
                                                                    *batch_embeddings.shape[-2:]).to("cpu")
                    elif self.model_class in ['PolyEncoder', 'QAMatchModel', 'CMCModel']:
                        batch_embeddings = batch_embeddings.reshape(this_batch_size, candidate_num,
                                                                    batch_embeddings.shape[-1]).to("cpu")
                    else:
                        raise Exception("This is not supported!!")

                    pbar.update(this_batch_size)
                    now_query_count += query_step

                    # (1, candidate_num(, context_num), dim)
                    whole_candidate_embeddings.append(batch_embeddings)

                pbar.close()

                # (query_num, candidate_num(, context_num), dim)
                whole_candidate_embeddings = torch.cat(whole_candidate_embeddings, dim=0).to("cpu")
                torch.save(whole_candidate_embeddings, "./output/" + self.dataset_name + "_" + self.model_save_prefix +
                           self.model_class)

            torch.cuda.empty_cache()
            gc.collect()
            print(f"Query num: {query_num}, Block size: {self.query_block_size}, {whole_candidate_embeddings.shape}")

            for i in range(0, 6):
                # begin time
                begin_time = time.time()

                whole_dot_products = []
                # calculate queries with or without candidates
                processed_query_count = 0

                # query_num = self.query_block_size

                # do match block by block
                this_batch_time = time.time()
                all_load = 0.0
                all_gpu = 0.0
                while processed_query_count < query_num:
                    this_block_query_input_ids = query_input_ids[
                                                 processed_query_count:processed_query_count + self.query_block_size,
                                                 :].to(self.device)
                    this_block_query_attention_mask = query_attention_mask[
                                                      processed_query_count:processed_query_count + self.query_block_size,
                                                      :].to(self.device)
                    this_block_query_token_type_ids = query_token_type_ids[
                                                      processed_query_count:processed_query_count + self.query_block_size,
                                                      :].to(self.device)
                    # get corresponding candidates
                    this_block_candidate_embeddings = whole_candidate_embeddings[
                                                      processed_query_count:processed_query_count + self.query_block_size].to(
                        self.device)

                    elapse_time = time.time() - this_batch_time
                    all_load += elapse_time
                    # print(f" Load data takes", get_elapse_time(elapse_time))
                    loaded_time = time.time()

                    if self.model_class in ['MatchDeformer', 'ColBERT']:
                        this_block_candidate_attention_mask = candidate_attention_mask[
                                                              processed_query_count:processed_query_count + self.query_block_size].to(
                            self.device)
                        dot_products = self.model.do_queries_match(input_ids=this_block_query_input_ids,
                                                                   token_type_ids=this_block_query_token_type_ids,
                                                                   attention_mask=this_block_query_attention_mask,
                                                                   candidate_context_embeddings=this_block_candidate_embeddings,
                                                                   candidate_attention_mask=this_block_candidate_attention_mask)
                    else:
                        dot_products = self.model.do_queries_match(input_ids=this_block_query_input_ids,
                                                                   token_type_ids=this_block_query_token_type_ids,
                                                                   attention_mask=this_block_query_attention_mask,
                                                                   candidate_context_embeddings=this_block_candidate_embeddings)

                    elapse_time = time.time() - loaded_time
                    all_gpu += elapse_time
                    # print(f" Calculate tasks", get_elapse_time(elapse_time))

                    whole_dot_products.append(dot_products)
                    processed_query_count += self.query_block_size

                    this_batch_time = time.time()

                whole_dot_products = torch.cat(whole_dot_products, dim=0)

                print(f"{whole_dot_products.shape}")
                elapse_time = time.time() - begin_time
                print(f"Model is {self.model_class}, Real scene testing takes", get_elapse_time(elapse_time))
                print(f"All load is {all_load}, All gpu is {all_gpu}, all {all_load + all_gpu}")
                print(whole_dot_products.shape)
                print("*" * 100)

    def match_bi_real_test_for_match_cross(self, test_datasets: DoubleInputDataset = None, model_save_path=None, **kwargs):
        """

        :param test_datasets: contain queries: (query_num, query_seq_len), contain candidates: (query_num, val_candidate_num, candidate_seq_len)
        :param model_save_path: use to load model if model is not created
        :param kwargs: placeholder
        :return: None
        """
        print("*" * 40 + " Begin Real Testing " + "*" * 40)

        # prepare data
        if os.path.exists(root_path + "dataset/" + self.dataset_name + "_real"):
            dataset = torch.load(root_path + "dataset/" + self.dataset_name + "_real")

            candidate_input_ids = dataset['candidate_input_ids']
            candidate_attention_mask = dataset['candidate_attention_mask']
            candidate_token_type_ids = dataset['candidate_token_type_ids']
            query_input_ids = dataset['query_input_ids']
            query_attention_mask = dataset['query_attention_mask']
            query_token_type_ids = dataset['query_token_type_ids']

            start_index = 0
            end_index = 100
            # (query_num, candidate_num, seq_len)
            # dstc: (1000, 100, 327)
            candidate_input_ids = candidate_input_ids[start_index:end_index]
            candidate_attention_mask = candidate_attention_mask[start_index:end_index]
            candidate_token_type_ids = candidate_token_type_ids[start_index:end_index]

            query_input_ids = query_input_ids[start_index:end_index]
            query_attention_mask = query_attention_mask[start_index:end_index]
            query_token_type_ids = query_token_type_ids[start_index:end_index]
        else:
            if not test_datasets:
                if self.dataset_name in ['msmarco', 'hard_msmarco']:
                    _, this_datasets, _ = self.__get_datasets(get_train=False, get_test=False)
                else:
                    _, _, this_datasets = self.__get_datasets(get_train=False, get_val=False)
            else:
                this_datasets = test_datasets

            if len(this_datasets) > 1:
                raise Exception("Test Datasets is more than 1 for match task!")

            this_datasets = this_datasets[0]

            start_index = 0
            end_index = 800
            # (query_num, candidate_num, seq_len)
            # dstc: (1000, 100, 327)
            candidate_input_ids = this_datasets.b_input_ids[start_index:end_index]
            candidate_attention_mask = this_datasets.b_attention_mask[start_index:end_index]
            candidate_token_type_ids = this_datasets.b_token_type_ids[start_index:end_index]

            query_input_ids = this_datasets.a_input_ids[start_index:end_index]
            query_attention_mask = this_datasets.a_attention_mask[start_index:end_index]
            query_token_type_ids = this_datasets.a_token_type_ids[start_index:end_index]

            print(candidate_input_ids.shape)

            torch.save(
                {"candidate_input_ids": candidate_input_ids, "candidate_attention_mask": candidate_attention_mask,
                 "candidate_token_type_ids": candidate_token_type_ids,
                 "query_input_ids": query_input_ids, "query_attention_mask": query_attention_mask,
                 "query_token_type_ids": query_token_type_ids}, root_path + "dataset/" + self.dataset_name + "_real")

        query_num = candidate_input_ids.shape[0]
        candidate_num = candidate_input_ids.shape[1]

        print(f"Query shape {query_input_ids.shape}, Candidate shape {candidate_input_ids.shape}")

        # create model if necessary
        if self.model is None:
            self.model = self.__create_model()
            # self.model = self.load_model(self.model, model_save_path)
            self.model.to(self.device)

        self.model.eval()

        for i in range(6):
            # pre-compute candidates
            with torch.no_grad():
                # begin time
                print("begin real test!!!!")
                whole_dot_products = []
                # calculate queries with or without candidates
                processed_query_count = 0

                # begin time
                begin_time = time.time()

                # do match block by block
                this_batch_time = time.time()
                all_load = 0.0
                all_gpu = 0.0

                query_num = 10

                # do match block by block
                while processed_query_count < query_num:
                    this_block_query_input_ids = query_input_ids[
                                                 processed_query_count:processed_query_count + self.query_block_size,
                                                 :].to(self.device)
                    this_block_query_attention_mask = query_attention_mask[
                                                      processed_query_count:processed_query_count + self.query_block_size,
                                                      :].to(self.device)
                    this_block_query_token_type_ids = query_token_type_ids[
                                                      processed_query_count:processed_query_count + self.query_block_size,
                                                      :].to(self.device)

                    this_block_candidate_input_ids = candidate_input_ids[
                                                     processed_query_count:processed_query_count + self.query_block_size,
                                                     :].to(self.device)
                    this_block_candidate_attention_mask = candidate_attention_mask[
                                                          processed_query_count:processed_query_count + self.query_block_size,
                                                          :].to(self.device)
                    this_block_candidate_token_type_ids = candidate_token_type_ids[
                                                          processed_query_count:processed_query_count + self.query_block_size,
                                                          :].to(self.device)

                    elapse_time = time.time() - this_batch_time
                    all_load += elapse_time
                    # print(f" Load data takes", get_elapse_time(elapse_time))
                    loaded_time = time.time()

                    dot_products = self.model(a_input_ids=this_block_query_input_ids,
                                              a_token_type_ids=this_block_query_token_type_ids,
                                              a_attention_mask=this_block_query_attention_mask,
                                              b_input_ids=this_block_candidate_input_ids,
                                              b_token_type_ids=this_block_candidate_token_type_ids,
                                              b_attention_mask=this_block_candidate_attention_mask,
                                              train_flag=False)

                    whole_dot_products.append(dot_products)
                    processed_query_count += self.query_block_size

                    elapse_time = time.time() - loaded_time
                    all_gpu += elapse_time
                    this_batch_time = time.time()

                whole_dot_products = torch.cat(whole_dot_products, dim=0)

                print(f"{whole_dot_products.shape}")
                elapse_time = time.time() - begin_time
                print(f"Model is {self.model_class}, Real scene testing takes", get_elapse_time(elapse_time))
                print(f"All load is {all_load}, All gpu is {all_gpu}, all {all_load + all_gpu}")
                print(whole_dot_products.shape)
                print("*" * 100)

    def match_cross_real_test(self, test_datasets: SingleInputDataset = None, model_save_path=None, **kwargs):
        print("*" * 40 + " Begin Real Testing " + "*" * 40)

        # prepare data
        if not test_datasets:
            _, _, this_datasets = self.__get_datasets(get_train=False, get_val=False)
        else:
            this_datasets = test_datasets
        if len(this_datasets) > 1:
            raise Exception("Test Datasets is more than 1 for match task!")

        val_dataloader = self.__get_dataloader(data=this_datasets, batch_size=self.val_batch_size)
        val_dataloader = val_dataloader[0]

        if self.model is None:
            self.model = self.__create_model()
            # self.model = self.load_model(self.model, model_save_path)
            self.model.to(self.device)

        self.model.eval()

        # begin time
        begin_time = time.time()

        with torch.no_grad():
            whole_logits = []
            for index, batch in enumerate(tqdm(val_dataloader)):
                input_ids = batch['input_ids'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                batch_num, candidate_num, sequence_len = input_ids.shape

                input_ids = input_ids.reshape(batch_num * candidate_num, -1)
                token_type_ids = token_type_ids.reshape(batch_num * candidate_num, -1)
                attention_mask = attention_mask.reshape(batch_num * candidate_num, -1)

                # 得到模型的结果
                logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                                    attention_mask=attention_mask).reshape(batch_num, candidate_num)
                whole_logits.append(logits)

            whole_logits = torch.cat(whole_logits, dim=0)
            # calculate R@K
            r_k_result = ()

            for k in self.r_k_num:
                _, indices = torch.topk(whole_logits, k, dim=-1)
                hit_num = ((indices == (self.val_candidate_num - 1)).sum(-1)).sum().item()
                r_k_result += (hit_num / whole_logits.shape[0],)
                print(f"R@" + str(k) + f": {r_k_result[-1]}", end="\t")

            _, avg_r_k_result = sum_average_tuple(r_k_result)
            print(f"Avg: {avg_r_k_result}")
            print(f"Query Num is {whole_logits.shape[0]}, Candidate Num is {candidate_num}")
            print(f"Model is {self.model_class}, Real scene testing takes", get_elapse_time(begin_time))
            print("*" * 100)

    def classify_bi_real_test(self, test_datasets: DoubleInputLabelDataset = None, model_save_path=None, **kwargs):
        print("*" * 40 + " Begin Real Testing " + "*" * 40)

        # prepare data
        if not test_datasets:
                _, _, this_datasets = self.__get_datasets(get_train=False, get_val=False)
        else:
            this_datasets = test_datasets

        this_datasets = this_datasets[0]

        # (query_num, candidate_num, seq_len)
        # mnli: (9796, 55)
        candidate_input_ids = this_datasets.b_input_ids
        candidate_attention_mask = this_datasets.b_attention_mask
        candidate_token_type_ids = this_datasets.b_token_type_ids

        # mnli: [9796, 234]
        query_input_ids = this_datasets.a_input_ids
        query_attention_mask = this_datasets.a_attention_mask
        query_token_type_ids = this_datasets.a_token_type_ids

        print(f"Query shape {query_input_ids.shape}, Candidate shape {candidate_input_ids.shape}")

        qa_labels = this_datasets.label
        # create model if necessary
        if self.model is None:
            self.model = self.__create_model()
            # self.model = self.load_model(self.model, model_save_path)
            self.model.to(self.device)

        self.model.eval()

        # split data because some model use too much memory
        query_num = candidate_input_ids.shape[0] // self.real_test_data_split_num

        candidate_input_ids = candidate_input_ids[:query_num]
        candidate_attention_mask = candidate_attention_mask[:query_num]
        candidate_token_type_ids = candidate_token_type_ids[:query_num]

        query_input_ids = query_input_ids[:query_num]
        query_attention_mask = query_attention_mask[:query_num]
        query_token_type_ids = query_token_type_ids[:query_num]

        # pre-compute candidates
        with torch.no_grad():
            candidate_num = candidate_input_ids.shape[0]
            processed_candidate_count = 0
            encoding_batch_size = 800

            # encoding one by one
            whole_candidate_embeddings = []
            while processed_candidate_count < candidate_num:
                print(f"\r{processed_candidate_count}/{candidate_num}", end="")
                # calculate vectors of candidates
                batch_input_ids = candidate_input_ids[
                                  processed_candidate_count:processed_candidate_count + encoding_batch_size].to(
                    self.device)
                batch_attention_mask = candidate_attention_mask[
                                       processed_candidate_count:processed_candidate_count + encoding_batch_size].to(
                    self.device)
                batch_token_type_ids = candidate_token_type_ids[
                                       processed_candidate_count:processed_candidate_count + encoding_batch_size].to(
                    self.device)

                batch_embeddings = self.model.prepare_candidates(input_ids=batch_input_ids,
                                                                 token_type_ids=batch_token_type_ids,
                                                                 attention_mask=batch_attention_mask)

                if self.model_class in ['ClassifyDeformer'] or self.dataset_name in ['qqp', 'boolq']:
                    whole_candidate_embeddings.append(batch_embeddings.to("cpu"))
                else:
                    whole_candidate_embeddings.append(batch_embeddings)

                processed_candidate_count += encoding_batch_size

            print()
            # (candidate_num(, context_num), dim)
            whole_candidate_embeddings = torch.cat(whole_candidate_embeddings, dim=0).to("cpu")
            print("candidate computed finished!")

            # begin time
            begin_time = time.time()

            whole_logits = []
            # calculate queries with or without candidates
            processed_query_count = 0
            query_num = query_input_ids.shape[0]
            # do match block by block
            while processed_query_count < query_num:
                this_block_query_input_ids = query_input_ids[
                                             processed_query_count:processed_query_count + self.query_block_size,
                                             :].to(self.device)
                this_block_query_attention_mask = query_attention_mask[
                                                  processed_query_count:processed_query_count + self.query_block_size,
                                                  :].to(self.device)
                this_block_query_token_type_ids = query_token_type_ids[
                                                  processed_query_count:processed_query_count + self.query_block_size,
                                                  :].to(self.device)
                # get corresponding candidates
                this_block_candidate_embeddings = whole_candidate_embeddings[
                                                  processed_query_count:processed_query_count + self.query_block_size].to(
                    self.device)
                this_block_candidate_attention_mask = candidate_attention_mask[
                                                      processed_query_count:processed_query_count + self.query_block_size].to(
                    self.device)

                logits = self.model.do_queries_classify(input_ids=this_block_query_input_ids,
                                                        token_type_ids=this_block_query_token_type_ids,
                                                        attention_mask=this_block_query_attention_mask,
                                                        candidate_context_embeddings=this_block_candidate_embeddings,
                                                        candidate_attention_mask=this_block_candidate_attention_mask)
                whole_logits.append(logits)
                processed_query_count += self.query_block_size

            whole_logits = torch.cat(whole_logits, dim=0)

            # 统计命中率
            label_target_num, label_shoot_num, label_hit_num = [], [], []
            for i in range(0, self.label_num):
                label_target_num.append((qa_labels == i).sum().item())
                label_shoot_num.append(0)
                label_hit_num.append(0)

            _, row_max_indices = whole_logits.topk(k=1, dim=-1)

            for i, max_index in enumerate(row_max_indices):
                inner_index = max_index[0]
                label_shoot_num[inner_index] += 1
                if inner_index == qa_labels[i]:
                    label_hit_num[inner_index] += 1

            accuracy = print_recall_precise(label_hit_num=label_hit_num, label_shoot_num=label_shoot_num,
                                            label_target_num=label_target_num, label_num=self.label_num)

            print(f"Avg: {accuracy}")
            print(f"Query Num is {query_num}, Candidate Num is {candidate_num}")
            print(f"Model is {self.model_class}, Real scene testing takes", get_elapse_time(begin_time))
            print("*" * 100)

    def classify_cross_real_test(self, test_datasets: SingleInputLabelDataset = None, model_save_path=None, **kwargs):
        print("*" * 40 + " Begin Real Testing " + "*" * 40)

        # prepare data
        if not test_datasets:
            _, _, this_datasets = self.__get_datasets(get_train=False, get_val=False)
        else:
            this_datasets = test_datasets

        val_dataloader = self.__get_dataloader(data=this_datasets, batch_size=self.val_batch_size)
        val_dataloader = val_dataloader[0]

        if self.model is None:
            self.model = self.__create_model()
            # self.model = self.load_model(self.model, model_save_path)
            self.model.to(self.device)

        self.model.eval()

        # begin time
        begin_time = time.time()

        with torch.no_grad():
            whole_logits = []
            # split data because some model use too much memory
            dataloader_len = len(val_dataloader) // self.real_test_data_split_num

            for index, batch in enumerate(tqdm(val_dataloader)):
                if index == dataloader_len:
                    break
                input_ids = batch['input_ids'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # 得到模型的结果
                logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)
                whole_logits.append(logits)

            whole_logits = torch.cat(whole_logits, dim=0)

            # 统计命中率
            qa_labels = this_datasets[0].label

            label_target_num, label_shoot_num, label_hit_num = [], [], []
            for i in range(0, self.label_num):
                label_target_num.append((qa_labels == i).sum().item())
                label_shoot_num.append(0)
                label_hit_num.append(0)

            _, row_max_indices = whole_logits.topk(k=1, dim=-1)

            for i, max_index in enumerate(row_max_indices):
                inner_index = max_index[0]
                label_shoot_num[inner_index] += 1
                if inner_index == qa_labels[i]:
                    label_hit_num[inner_index] += 1

            accuracy = print_recall_precise(label_hit_num=label_hit_num, label_shoot_num=label_shoot_num,
                                            label_target_num=label_target_num, label_num=self.label_num)

            print(f"Avg: {accuracy}")
            print(f"Model is {self.model_class}, Real scene testing takes", get_elapse_time(begin_time))
            print("*" * 100)

    def classify_do_val_body(self, val_datasets, previous_best_performance):
        val_dataloader = self.__get_dataloader(data=val_datasets, batch_size=self.val_batch_size)

        val_loss_tuple, val_acc_tuple = (), ()
        for dataloader in val_dataloader:
            this_val_loss, this_val_acc = self.classify_validate_model(dataloader)
            val_loss_tuple = val_loss_tuple + (this_val_loss,)
            val_acc_tuple = val_acc_tuple + (this_val_acc,)

        sum_val_acc, avg_val_acc = sum_average_tuple(val_acc_tuple)
        sum_val_loss, avg_val_loss = sum_average_tuple(val_loss_tuple)

        print(
            f"Eval on Validation Dataset: " + "*" * 30 +
            f"\nval_loss:{avg_val_loss}\tval_acc:{avg_val_acc}%\tprevious best performance:{previous_best_performance}%\tfrom rank:{self.local_rank}")

        return avg_val_acc

    def match_val_test_body(self, previous_best_performance,  this_datasets=None, model_save_path=None, do_test=False, **kwargs):
        """
        calculate R@K, and loss
        :param model_save_path:
        :param this_datasets:
        :param do_test: do test or validation
        :param previous_best_performance: used to print
        :return: (loss, R@K, ...)
        """

        if not this_datasets:
            if not do_test:
                _, test_datasets, _ = self.__get_datasets(get_train=False, get_test=False)
            else:
                _, _, test_datasets = self.__get_datasets(get_train=False, get_val=False)
        else:
            test_datasets = this_datasets

        val_dataloader = self.__get_dataloader(data=test_datasets, batch_size=self.val_batch_size)

        # create model if necessary
        if self.model is None:
            self.model = self.__create_model()
            self.model.to(self.device)

            if APEX_FLAG and not self.no_apex:
                self.model = amp.initialize(self.model, opt_level="O1")

            if self.load_model_flag: # jcy
                self.model = self.load_model(self.model, self.load_model_path)
            else:    
                self.model = self.load_model(self.model, model_save_path)

        self.model.eval()

        if len(val_dataloader) > 1:
            raise Exception("Match val_dataloader is more than 1.")

        cross_entropy_function = nn.CrossEntropyLoss()
        this_loss, r_k_result = None, None

        # global statement
        avg_r_k_result = -999
        mrr_depth, mrr_avg = 0.0, 0.0

        for dataloader in val_dataloader:
            logits, q_ids, p_ids, actual_candidate_num = self.match_validate_model(dataloader)
            logits = logits.to(self.device)

            # read labels for marco topk
            if self.dataset_name in ["msmarco", "hard_msmarco"]:
                qrels = load_qrels(root_path + "dataset/qrels.dev.tsv")

                ranking_label = []
                for index, qid in enumerate(q_ids):
                    ranking_label.append(qrels[qid])
                    logits[index, actual_candidate_num[index]:] = -np.inf

                mrr_depth = 10
            else:
                ranking_label = self.val_candidate_num - 1
                mrr_depth = self.val_candidate_num

            # calculate R@K
            avg_r_k_result, r_k_result = calculate_recall(logits, self.r_k_num, ranking_label, p_ids=p_ids)

            # calculate MRR
            mrr_avg, mrr_sums = calculate_mrr(logits, mrr_depth, ranking_label, p_ids=p_ids)

            # calculate loss
            if not(self.dataset_name in ["msmarco", "hard_msmarco"]):
                my_label = torch.tensor([self.val_candidate_num-1]*logits.shape[0], dtype=torch.long, device=logits.device)
                this_loss = cross_entropy_function(logits, my_label) # jcy 
            else:
                this_loss = 0.0

        log_text = "test" if do_test else "validation"

        print(
            f"Eval on {log_text} Dataset: " + "*" * 30 +
            f"\nval_loss:{this_loss}\tR@K:{r_k_result}%\tR_k:{self.r_k_num}\t"
            f"Avg R:{avg_r_k_result}\tprevious best performance:{previous_best_performance}"
            f"\nMRR@{mrr_depth}:{mrr_avg}"
            f"\nfrom rank:{self.local_rank}")

        # in order to use > to reveal priority
        return mrr_avg

    def restore_settings(self, optimizer, restore_path, previous_best_performance):
        early_stop_count = 0
        restore_epoch = 0
        scheduler_last_epoch = 0
        new_previous_best_performance = previous_best_performance

        if self.restore_flag:
            restore_data = torch.load(restore_path)

            # get scheduler
            scheduler_last_epoch = restore_data['scheduler']['last_epoch']
            early_stop_count = restore_data['early_stop_count']

            # get optimizer
            optimizer.load_state_dict(restore_data['optimizer'])

            # to device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

            # get best performance
            new_previous_best_performance = restore_data['best performance']

            # get epoch
            restore_epoch = restore_data['epoch']

            print("model is restored from", restore_path)
            print(f'Restore epoch: {restore_epoch}, Previous best performance: {new_previous_best_performance}')
            print("*" * 100)

        return early_stop_count, restore_epoch, scheduler_last_epoch, new_previous_best_performance

    def get_restore_path(self, final_stage_flag):
        restore_path = self.last_model_dict + self.model_save_prefix + self.model_class + "_" + self.dataset_name
        if not final_stage_flag:
            restore_path += "_middle"
        return restore_path

    # classify
    def classify_validate_model(self, dataloader):
        self.model.eval()

        label_target_num = [0] * self.label_num
        label_shoot_num = [0] * self.label_num
        label_hit_num = [0] * self.label_num

        # 开始评测
        with torch.no_grad():
            val_loss = 0.0
            cross_entropy_function = nn.CrossEntropyLoss()

            # 生成dataloader
            classify_dataloader = dataloader

            print(f"------- begin val {self.val_batch_size * len(classify_dataloader)} data--------")

            # 计算正确率
            shoot_num = 0
            hit_num = 0
            for index, batch in enumerate(classify_dataloader):
                # 读取数据
                # add model
                if self.model_class in ['QAClassifierModel', 'ClassifyParallelEncoder', 'PolyEncoder',
                                        'ClassifyDeformer', 'CLSClassifyParallelEncoder',
                                        'DisenCLSClassifyParallelEncoder', 'DisenClassifyParallelEncoder']:
                    logits = self.__val_step_for_qa_input(batch)
                elif self.model_class in ['CrossBERT']:
                    logits = self.__classify_val_step_for_cross(batch)
                else:
                    raise Exception("Val step is not supported for this model class!")

                qa_labels = (batch['label']).to(self.device)

                if index == 0:
                    print(f"First logits during validate: {logits[0]}\t Its label is {qa_labels[0]}")
                    print(f"Second logits during validate: {logits[1]}\t Its label is {qa_labels[1]}")
                    print(f"Third logits during validate: {logits[2]}\t Its label is {qa_labels[2]}")

                loss = cross_entropy_function(logits, qa_labels)
                val_loss += loss.item()

                # 统计命中率
                shoot_num += len(qa_labels)
                for i in range(0, self.label_num):
                    label_target_num[i] += (qa_labels == i).sum().item()

                batch_hit_num = 0
                _, row_max_indices = logits.topk(k=1, dim=-1)

                for i, max_index in enumerate(row_max_indices):
                    inner_index = max_index[0]
                    label_shoot_num[inner_index] += 1
                    if inner_index == qa_labels[i]:
                        label_hit_num[inner_index] += 1
                        batch_hit_num += 1
                hit_num += batch_hit_num

            accuracy = print_recall_precise(label_hit_num=label_hit_num, label_shoot_num=label_shoot_num,
                                            label_target_num=label_target_num, label_num=self.label_num)

            self.model.train()

        return val_loss, accuracy

    def match_validate_model(self, dataloader):
        """
        :param dataloader: (query_num, candidate_num, seq_len)
        :return: logits = (query_num, candidate_num)
        """
        self.model.eval()

        # 开始评测
        with torch.no_grad():
            # 生成dataloader
            match_dataloader = dataloader

            print(f"------- begin val {self.val_batch_size * len(match_dataloader)} data--------")

            # 计算正确率
            whole_logits = []
            q_ids, p_ids, actual_candidate_num = [], [], []

            for index, batch in enumerate(tqdm(match_dataloader)):
                if self.dataset_name in ['msmarco', "hard_msmarco"]:
                    q_ids += batch['idx'].tolist()
                    actual_candidate_num += batch['candidate_num'].tolist()
                    p_ids += batch['pid'].tolist()

                # 读取数据
                # add model
                if self.model_class in ['QAMatchModel', 'CMCModel', 'MatchParallelEncoder', 'PolyEncoder', 'MatchDeformer',
                                        'MatchCrossBERT', 'CLSMatchParallelEncoder', 'DisenMatchParallelEncoder',
                                        'DisenCLSMatchParallelEncoder', 'ColBERT']:
                    logits = self.__match_val_step_for_bi(batch)
                elif self.model_class in ['CrossBERT']:
                    logits = self.__match_val_step_for_cross(batch)
                else:
                    raise Exception("match_validate_model is not supported for this model class!")

                whole_logits.append(logits.to("cpu"))

            whole_logits = torch.cat(whole_logits, dim=0)

        return whole_logits, q_ids, p_ids, actual_candidate_num

    def glue_test(self, test_datasets=None, model_save_path=None, postfix=None, do_val=False):
        # add dataset
        dataset_label_dict = {'mnli': ['entailment', 'neutral', 'contradiction'],
                              'qqp': ['0', '1'], 'boolq': ['false', 'true']}
        output_text_name = {'mnli': ['MNLI-m.tsv', 'MNLI-mm.tsv'], 'qqp': ['QQP.tsv'], 'boolq': ['BoolQ.jsonl']}

        # create model if necessary
        if self.model is None:
            self.model = self.__create_model()
            self.model.to(self.device)

            if APEX_FLAG and not self.no_apex:
                self.model = amp.initialize(self.model, opt_level="O1")

            self.model = self.load_model(self.model, model_save_path)

        self.model.eval()

        # read datasets if necessary
        if test_datasets is None:
            if not do_val:
                _, _, test_datasets = self.__get_datasets()
            else:
                _, test_datasets, _ = self.__get_datasets()
                self.classify_do_val_body(test_datasets, 0.0)
                return None

        test_dataloaders = self.__get_dataloader(data=test_datasets, batch_size=self.val_batch_size)

        for dataloader_index, this_dataloader in enumerate(test_dataloaders):
            all_prediction = []

            # 开始评测
            with torch.no_grad():
                print(f"------- begin test {self.val_batch_size * len(this_dataloader)} data--------")

                bar = tqdm(this_dataloader, total=len(this_dataloader))

                for index, batch in enumerate(bar):
                    this_batch_index = batch['idx']

                    # 读取数据
                    # add model
                    if self.model_class in ['QAClassifierModel', 'ClassifyParallelEncoder', 'PolyEncoder',
                                            'ClassifyDeformer', 'CLSClassifyParallelEncoder',
                                            'DisenCLSClassifyParallelEncoder', 'DisenClassifyParallelEncoder']:
                        logits = self.__val_step_for_qa_input(batch)
                    elif self.model_class in ['CrossBERT']:
                        logits = self.__classify_val_step_for_cross(batch)
                    else:
                        raise Exception("Val step is not supported for this model class!")

                    _, row_max_indices = logits.topk(k=1, dim=-1)
                    this_prediction = [(this_batch_index[i], item[0]) for i, item in enumerate(row_max_indices)]
                    all_prediction += this_prediction

                # write to disk
                if not os.path.exists("./output/"):
                    os.makedirs("./output/")

                # use jsonl
                if self.dataset_name in ['boolq']:
                    with jsonlines.open(
                        "./output/" + self.model_save_prefix + self.model_class + output_text_name[self.dataset_name][
                            dataloader_index] + postfix, "w") as writer:
                        for (index, pre) in all_prediction:
                            writer.write({"idx":index.item(), "label":dataset_label_dict[self.dataset_name][pre]})
                # use tsv
                else:
                    with open("./output/" + self.model_save_prefix + self.model_class + output_text_name[self.dataset_name][dataloader_index] + postfix, "w") as writer:
                        tsv_writer = csv.writer(writer, delimiter='\t', lineterminator='\n')

                        tsv_writer.writerow(['index', 'prediction'])
                        for (index, pre) in all_prediction:
                            text_pre = dataset_label_dict[self.dataset_name][pre]
                            tsv_writer.writerow([index.item(), text_pre])

                print(f"Result is saved to ./output/{self.model_save_prefix + self.model_class + output_text_name[self.dataset_name][dataloader_index] + postfix}")

        self.model.train()

    def save_model(self, model_save_path, optimizer, scheduler, epoch, previous_best_performance, early_stop_count):
        self.model.eval()

        # Only save the model it-self, maybe parallel
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        # 保存模型
        save_state = {'pretrained_bert_path': self.config.pretrained_bert_path,
                      'model': model_to_save.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict(),
                      'best performance': previous_best_performance,
                      'early_stop_count': early_stop_count,
                      'epoch': epoch + 1}

        if APEX_FLAG and not self.no_apex:
            save_state['amp'] = amp.state_dict()

        torch.save(save_state, model_save_path)

        print("!" * 60)
        print(f"model is saved at {model_save_path}")
        print("!" * 60)

        self.model.train()

    def load_model(self, model, load_model_path):
        if load_model_path is None:
            print("you should offer model paths!")

        load_path = load_model_path
        checkpoint = torch.load(load_path)

        model.load_state_dict(checkpoint['model'])
        if APEX_FLAG and not self.no_apex:
            print("make sure amp.initialize before load model!!!!")
            try:
                amp.load_state_dict(checkpoint['amp'])
            except KeyError:
                print("!"*100)
                print("Loading a checkpoint without amp!!!!")
                print("!"*100)

        # if self.data_distribute:
        # 	load_path += ".pt"
        # 	checkpoint = torch.load(load_path)
        # 	self.model.module.load_state_dict(checkpoint['model'])
        # 	amp.load_state_dict(checkpoint['amp'])
        # elif self.data_parallel:
        # 	load_path += ".pt"
        # 	checkpoint = torch.load(load_path)
        # 	self.model.module.load_state_dict(checkpoint['model'])
        # else:
        # 	self.model.load_state_dict(torch.load(load_path))

        print("model is loaded from", load_path)

        return model

    # return must be tuple
    def __get_dataloader(self, data, batch_size, train_flag=False):
        shuffle_flag = drop_last_flag = train_flag

        all_dataloader = ()

        for now_data in data:
            if self.data_distribute:
                sampler = torch.utils.data.distributed.DistributedSampler(now_data, shuffle=shuffle_flag, drop_last=drop_last_flag)
                dataloader = DataLoader(now_data, batch_size=batch_size, sampler=sampler)
            else:
                dataloader = DataLoader(now_data, batch_size=batch_size, shuffle=shuffle_flag,
                                        drop_last=drop_last_flag)

            all_dataloader = all_dataloader + (dataloader, )

        return all_dataloader

    def __get_datasets(self, get_train=True, get_val=True, get_test=True):
        # prepare dataset
        # must be tuple
        train_datasets = ()
        val_datasets = ()
        test_datasets = ()

        # add model
        # Checking whether input is pair or single is important
        if self.model_class in ['QAClassifierModel', 'ClassifyParallelEncoder', 'PolyEncoder', 'QAMatchModel', 'CMCModel',
                                'MatchParallelEncoder', 'ClassifyDeformer', 'MatchDeformer', 'MatchCrossBERT',
                                'CLSMatchParallelEncoder', 'CLSClassifyParallelEncoder', 'DisenMatchParallelEncoder',
                                'DisenCLSMatchParallelEncoder', 'DisenCLSClassifyParallelEncoder',
                                'DisenClassifyParallelEncoder', 'ColBERT']:
            pair_flag = True
            save_load_prefix = ""
            save_load_suffix = ""
        elif self.model_class in ['CrossBERT']:
            pair_flag = False
            save_load_prefix = "cross_"
            save_load_suffix = "_" + str(self.train_candidate_num)
        # elif self.model_class == 'CMCModel':
        #     pair_flag = True
        #     save_load_prefix = "cmc_"
        #     save_load_suffix = ""
        else:
            raise_dataset_error()

        if self.model_class in ['ClassifyDeformer']:
            save_load_prefix = "deformer_" + str(self.first_seq_max_len) + "_" + save_load_prefix

        # add text_max_len info to save_load_prefix
        if self.text_max_len != 512:
            save_load_prefix = str(self.text_max_len) + "_" + save_load_prefix

        # add dataset
        if self.dataset_name == 'mnli':
            # choose function to process data
            if pair_flag is True:
                temp_dataset_process_function = self.__tokenize_classify_bi_data_then_save
            else:
                temp_dataset_process_function = self.__tokenize_classify_cross_data_then_save

            # have been processed and saved to disk
            if os.path.exists(root_path + "dataset/" + save_load_prefix + "glue_mnli_train"):
                train_datasets = (
                torch.load(root_path + "dataset/" + save_load_prefix + "glue_mnli_train")['dataset'],)

                val_datasets = (torch.load(root_path + "dataset/" + save_load_prefix + "glue_mnli_val_matched")['dataset'],
                                torch.load(root_path + "dataset/" + save_load_prefix + "glue_mnli_val_mismatched")['dataset'],)

                test_datasets = (torch.load(root_path + "dataset/" + save_load_prefix + "glue_mnli_test_matched")['dataset'],
                                torch.load(root_path + "dataset/" + save_load_prefix + "glue_mnli_test_mismatched")['dataset'],)
            else:
                # load data from huggingface(online)
                complete_dataset = datasets.load_dataset("glue", 'mnli')

                # get train dataset
                train_datasets = (temp_dataset_process_function(data=complete_dataset['train'],
                                                                save_name=save_load_prefix + "glue_mnli_train",
                                                                a_column_name="premise",
                                                                b_column_name="hypothesis",
                                                                label_column_name='label'),)

                # get val dataset
                validation_matched_dataset = temp_dataset_process_function(data=complete_dataset['validation_matched'],
                                                                           save_name=save_load_prefix + "glue_mnli_val_matched",
                                                                           a_column_name="premise",
                                                                           b_column_name="hypothesis",
                                                                           label_column_name='label')
                validation_mismatched_dataset = temp_dataset_process_function(
                    data=complete_dataset['validation_mismatched'],
                    save_name=save_load_prefix + "glue_mnli_val_mismatched",
                    a_column_name="premise",
                    b_column_name="hypothesis",
                    label_column_name='label')

                val_datasets = (validation_matched_dataset, validation_mismatched_dataset,)

                # get test dataset
                test_matched_dataset = temp_dataset_process_function(data=complete_dataset['test_matched'],
                                                                     save_name=save_load_prefix + "glue_mnli_test_matched",
                                                                     a_column_name="premise",
                                                                     b_column_name="hypothesis",
                                                                     label_column_name='label')
                test_mismatched_dataset = temp_dataset_process_function(
                    data=complete_dataset['test_mismatched'],
                    save_name=save_load_prefix + "glue_mnli_test_mismatched",
                    a_column_name="premise",
                    b_column_name="hypothesis",
                    label_column_name='label')

                test_datasets = (test_matched_dataset, test_mismatched_dataset,)
        elif self.dataset_name == 'qqp':
            # choose function to process data
            if pair_flag is True:
                temp_dataset_process_function = self.__tokenize_classify_bi_data_then_save
            else:
                temp_dataset_process_function = self.__tokenize_classify_cross_data_then_save

            # have been processed and saved to disk
            if os.path.exists(root_path + "dataset/" + save_load_prefix + "glue_qqp_train"):
                train_datasets = (
                torch.load(root_path + "dataset/" + save_load_prefix + "glue_qqp_train")['dataset'],)

                val_datasets = (torch.load(root_path + "dataset/" + save_load_prefix + "glue_qqp_val")['dataset'],)

                test_datasets = (torch.load(root_path + "dataset/" + save_load_prefix + "glue_qqp_test")['dataset'],)
            else:
                # load data from huggingface(online)
                complete_dataset = datasets.load_dataset("glue", 'qqp')

                # get train dataset
                train_datasets = (temp_dataset_process_function(data=complete_dataset['train'],
                                                                save_name=save_load_prefix + "glue_qqp_train",
                                                                a_column_name="question1",
                                                                b_column_name="question2",
                                                                label_column_name='label'),)

                # get val dataset
                validation_dataset = temp_dataset_process_function(data=complete_dataset['validation'],
                                                                   save_name=save_load_prefix + "glue_qqp_val",
                                                                   a_column_name="question1",
                                                                   b_column_name="question2",
                                                                   label_column_name='label')

                val_datasets = (validation_dataset,)

                # get test dataset
                test_dataset = temp_dataset_process_function(data=complete_dataset['test'],
                                                             save_name=save_load_prefix + "glue_qqp_test",
                                                             a_column_name="question1",
                                                             b_column_name="question2",
                                                             label_column_name='label')
                test_datasets = (test_dataset,)
        elif self.dataset_name == 'boolq':
            # choose function to process data
            if pair_flag is True:
                temp_dataset_process_function = self.__tokenize_classify_bi_data_then_save
            else:
                temp_dataset_process_function = self.__tokenize_classify_cross_data_then_save

            # have been processed and saved to disk
            if os.path.exists(root_path + "dataset/" + save_load_prefix + "glue_boolq_train"):
                train_datasets = (
                    torch.load(root_path + "dataset/" + save_load_prefix + "glue_boolq_train")['dataset'],)

                val_datasets = (torch.load(root_path + "dataset/" + save_load_prefix + "glue_boolq_val")['dataset'],)

                test_datasets = (torch.load(root_path + "dataset/" + save_load_prefix + "glue_boolq_test")['dataset'],)
            else:
                # load data from huggingface(online)
                complete_dataset = datasets.load_dataset("super_glue", 'boolq')

                # get train dataset
                train_datasets = (temp_dataset_process_function(data=complete_dataset['train'],
                                                                save_name=save_load_prefix + "glue_boolq_train",
                                                                a_column_name="question",
                                                                b_column_name="passage",
                                                                label_column_name='label'),)

                # get val dataset
                validation_dataset = temp_dataset_process_function(data=complete_dataset['validation'],
                                                                   save_name=save_load_prefix + "glue_boolq_val",
                                                                   a_column_name="question",
                                                                   b_column_name="passage",
                                                                   label_column_name='label')

                val_datasets = (validation_dataset,)

                # get test dataset
                test_dataset = temp_dataset_process_function(data=complete_dataset['test'],
                                                             save_name=save_load_prefix + "glue_boolq_test",
                                                             a_column_name="question",
                                                             b_column_name="passage",
                                                             label_column_name='label')
                test_datasets = (test_dataset,)
        elif self.dataset_name == 'yahooqa':
            # check
            if not pair_flag and self.train_candidate_num < 1:
                raise Exception("Should designate train_candidate_num if you want train cross model on match task!!")

            # read or process...
            # read training data. exist? read!--------------------------------------------------
            if not get_train:
                pass
            elif os.path.exists(root_path + "dataset/" + save_load_prefix + self.dataset_name + "_train" + save_load_suffix):
                train_datasets = (
                torch.load(root_path + "dataset/" + save_load_prefix + self.dataset_name + "_train" + save_load_suffix)[
                    'dataset'],)
            else:
                if pair_flag:
                    string_train_dataset = datasets.load_from_disk(
                        root_path + "dataset/string_train_" + self.dataset_name)

                    train_datasets = (self.__tokenize_match_multi_candidate_data_then_save(data=string_train_dataset,
                                                                                           save_name=save_load_prefix + self.dataset_name + "_train",
                                                                                           a_column_name="sentence_a",
                                                                                           b_column_name="candidates",
                                                                                           candidate_num=5),)
                else:
                    string_train_dataset = datasets.load_from_disk(
                        root_path + "dataset/string_train_" + self.dataset_name)

                    train_datasets = (self.__tokenize_match_cross_data_then_save(data=string_train_dataset,
                                                                                 save_name=save_load_prefix + self.dataset_name + "_train",
                                                                                 a_column_name="sentence_a",
                                                                                 b_column_name="candidates",
                                                                                 candidate_num=5,
                                                                                 suffix="_5"),)

            # read val data. exist? read!--------------------------------------------------
            if pair_flag:
                process_val_test_func = self.__tokenize_match_multi_candidate_data_then_save
            else:
                process_val_test_func = self.__tokenize_match_cross_data_then_save

            if not get_val:
                pass
            elif os.path.exists(root_path + "dataset/" + save_load_prefix + self.dataset_name + "_val"):
                val_datasets = (torch.load(root_path + "dataset/" + save_load_prefix + self.dataset_name + "_val")['dataset'],)
            else:
                string_val_dataset = datasets.load_from_disk(root_path + "dataset/string_val_" + self.dataset_name)

                val_datasets = (process_val_test_func(data=string_val_dataset,
                                                      save_name=save_load_prefix + self.dataset_name + "_val",
                                                      a_column_name="sentence_a",
                                                      b_column_name="candidates",
                                                      candidate_num=5),)

            # read test data. exist? read!--------------------------------------------------
            if not get_test:
                pass
            elif os.path.exists(root_path + "dataset/" + save_load_prefix + self.dataset_name + "_test"):
                test_datasets = (torch.load(root_path + "dataset/" + save_load_prefix + self.dataset_name + "_test")['dataset'],)
            else:
                string_test_dataset = datasets.load_from_disk(root_path + "dataset/string_test_" + self.dataset_name)

                test_datasets = (process_val_test_func(data=string_test_dataset,
                                                       save_name=save_load_prefix + self.dataset_name + "_test",
                                                       a_column_name="sentence_a",
                                                       b_column_name="candidates",
                                                       candidate_num=5),)
        elif self.dataset_name == 'msmarco':
            # check
            if not pair_flag and self.train_candidate_num < 1:
                raise Exception("Should designate train_candidate_num if you want train cross model on match task!!")

            if self.in_batch:
                in_batch_suffix = "_in_batch_negative"
            else:
                in_batch_suffix = ""

            # read or process...
            # read training data. exist? read!--------------------------------------------------
            if not get_train:
                pass
            elif os.path.exists(root_path + "dataset/" + save_load_prefix + self.dataset_name + "_train" + in_batch_suffix):
                train_datasets = (
                torch.load(root_path + "dataset/" + save_load_prefix + self.dataset_name + "_train" + in_batch_suffix)[
                    'dataset'],)
            else:
                if pair_flag:
                    tokenized_query = torch.load(root_path + "dataset/msmarco_query_tokenized_dict")
                    tokenized_positive = torch.load(root_path + "dataset/msmarco_positive_tokenized_dict")

                    if self.in_batch: # jcy : I think it is false 
                        dataset = DoubleInputDataset(a_input_ids=tokenized_query['input_ids'],
                                                     a_attention_mask=tokenized_query['attention_mask'],
                                                     a_token_type_ids=tokenized_query['token_type_ids'],
                                                     b_input_ids=tokenized_positive['input_ids'],
                                                     b_token_type_ids=tokenized_positive['token_type_ids'],
                                                     b_attention_mask=tokenized_positive['attention_mask'],
                                                     idx=torch.tensor(list(range(tokenized_query['input_ids'].shape[0]))))
                    else:
                        tokenized_negative = torch.load(root_path + "dataset/msmarco_negative_tokenized_dict")

                        # concatenate candidates, put positive at last
                        positive_input_ids = tokenized_positive['input_ids'].unsqueeze(1) # (8, 1, 512)
                        positive_attention_mask = tokenized_positive['attention_mask'].unsqueeze(1)
                        positive_token_type_ids = tokenized_positive['token_type_ids'].unsqueeze(1)

                        negative_input_ids = tokenized_negative['input_ids'].unsqueeze(1) # (8, 1, 512)
                        negative_attention_mask = tokenized_negative['attention_mask'].unsqueeze(1)
                        negative_token_type_ids = tokenized_negative['token_type_ids'].unsqueeze(1)

                        candidate_input_ids = torch.cat((negative_input_ids, positive_input_ids), dim=1) # (8, 2, 512)
                        candidate_attention_mask = torch.cat((negative_attention_mask, positive_attention_mask), dim=1)
                        candidate_token_type_ids = torch.cat((negative_token_type_ids, positive_token_type_ids), dim=1)

                        dataset = DoubleInputDataset(a_input_ids=tokenized_query['input_ids'],
                                                     a_attention_mask=tokenized_query['attention_mask'],
                                                     a_token_type_ids=tokenized_query['token_type_ids'],
                                                     b_input_ids=candidate_input_ids,
                                                     b_token_type_ids=candidate_token_type_ids,
                                                     b_attention_mask=candidate_attention_mask,
                                                     idx=torch.tensor(list(range(tokenized_query['input_ids'].shape[0]))))

                    train_datasets = (dataset, )
                    if not os.path.exists(root_path + "dataset"):
                        os.makedirs(root_path + "dataset")

                    save_name = save_load_prefix + self.dataset_name + "_train"
                    if self.in_batch:
                        save_name += in_batch_suffix

                    torch.save({'dataset': dataset}, root_path + "dataset/" + save_name)

                    print(f"Processed dataset is saved at ./dataset/{save_name}")
                    print("*" * 20 + f"Encoding {tokenized_query['input_ids'].shape} texts finished!" + "*" * 20)
                else:
                    # directly copy from yahooqa, not check yet
                    raise Exception("Not support yet!")

            # read val data. exist? read!--------------------------------------------------
            if pair_flag:
                process_val_test_func = self.__tokenize_match_multi_candidate_data_then_save
            else:
                process_val_test_func = self.__tokenize_match_cross_data_then_save

            if not get_val:
                pass
            elif os.path.exists(root_path + "dataset/" + save_load_prefix + self.dataset_name + "_val"):
                val_datasets = (torch.load(root_path + "dataset/" + save_load_prefix + self.dataset_name + "_val")['dataset'],)
            else:
                string_val_dataset = torch.load(root_path + "dataset/string_msmarco_top1000_dev")

                val_datasets = (process_val_test_func(data=string_val_dataset,
                                                      save_name=save_load_prefix + self.dataset_name + "_val",
                                                      a_column_name="sentence_a",
                                                      b_column_name="candidates",
                                                      candidate_num=1000),)

            # read test data. exist? read!--------------------------------------------------
            # if not get_test:
            #     pass
            # elif os.path.exists(root_path + "dataset/" + save_load_prefix + self.dataset_name + "_test"):
            #     test_datasets = (torch.load(root_path + "dataset/" + save_load_prefix + self.dataset_name + "_test")['dataset'],)
            # else:
            #     string_test_dataset = datasets.load_from_disk(root_path + "dataset/string_test_" + self.dataset_name)
            #
            #     test_datasets = (process_val_test_func(data=string_test_dataset,
            #                                            save_name=save_load_prefix + self.dataset_name + "_test",
            #                                            a_column_name="sentence_a",
            #                                            b_column_name="candidates",
            #                                            candidate_num=1000),)
            test_datasets = None
        elif self.dataset_name == 'hard_msmarco': #TODO!
            # check
            if not pair_flag and self.train_candidate_num < 1:
                raise Exception("Should designate train_candidate_num if you want train cross model on match task!!")

            # read or process...
            # read training data. exist? read!--------------------------------------------------
            if not get_train:
                pass
            elif os.path.exists(root_path + "dataset/" + save_load_prefix + self.dataset_name + "_train"):
                print("Loading hard_msmarco_train from disk...")
                train_datasets = (
                torch.load(root_path + "dataset/" + save_load_prefix + self.dataset_name + "_train")[
                    'dataset'],)
            else: # hard_msmarco 
                print("Processing hard_msmarco_train...")
                if pair_flag:
                    tokenized_train_data_dict = torch.load(root_path + "dataset/tokenized_hard_msmarco_train")

                    dataset = MSMARCODataset(queries_ids=tokenized_train_data_dict['queries_ids'],
                                             qid_2_tensors=tokenized_train_data_dict['qid_2_tensors'],
                                             qid_2_negs=tokenized_train_data_dict['qid_2_negs'],
                                             qid_2_pos=tokenized_train_data_dict['qid_2_pos'],
                                             pid_2_tensors=tokenized_train_data_dict['pid_2_tensors'])

                    train_datasets = (dataset, )
                    if not os.path.exists(root_path + "dataset"):
                        os.makedirs(root_path + "dataset")

                    save_name = save_load_prefix + self.dataset_name + "_train"

                    torch.save({'dataset': dataset}, root_path + "dataset/" + save_name)

                    print(f"Processed dataset is saved at ./dataset/{save_name}")
                else:
                    # directly copy from yahooqa, not check yet
                    raise Exception("Not support yet!")

            # read val data. exist? read!--------------------------------------------------
            if pair_flag:
                process_val_test_func = self.__tokenize_match_multi_candidate_data_then_save
            else:
                process_val_test_func = self.__tokenize_match_cross_data_then_save

            if not get_val:
                pass
            elif os.path.exists(root_path + "dataset/" + save_load_prefix + "msmarco" + "_val"):
                val_datasets = (torch.load(root_path + "dataset/" + save_load_prefix + "msmarco" + "_val")['dataset'],)
            else:
                string_val_dataset = torch.load(root_path + "dataset/string_msmarco_top1000_dev")

                val_datasets = (process_val_test_func(data=string_val_dataset,
                                                      save_name=save_load_prefix + "msmarco" + "_val",
                                                      a_column_name="sentence_a",
                                                      b_column_name="candidates",
                                                      candidate_num=1000),)

            test_datasets = None
        elif self.dataset_name in ['dstc7', 'ubuntu']:
            # check
            if not pair_flag and self.train_candidate_num < 1:
                raise Exception("Should designate train_candidate_num if you want train cross model on match task!!")

            # read or process...
            # read training data. exist? read!--------------------------------------------------
            if not get_train:
                pass
            elif os.path.exists(root_path + "dataset/" + save_load_prefix + self.dataset_name + "_train" + save_load_suffix):
                train_datasets = (torch.load(root_path + "dataset/" + save_load_prefix + self.dataset_name + "_train" + save_load_suffix)['dataset'],)
            else:
                if pair_flag:
                    string_train_dataset = datasets.load_from_disk(
                        root_path + "dataset/string_bi_train_" + self.dataset_name)

                    train_datasets = (self.__tokenize_match_bi_data_then_save(data=string_train_dataset,
                                                                              save_name=save_load_prefix + self.dataset_name + "_train",
                                                                              a_column_name="sentence_a",
                                                                              b_column_name="sentence_b"),)
                else:
                    string_train_dataset = datasets.load_from_disk(
                        root_path + "dataset/string_cross_train_" + self.dataset_name)

                    train_datasets = (self.__tokenize_match_cross_data_then_save(data=string_train_dataset,
                                                                                 save_name=save_load_prefix + self.dataset_name + "_train",
                                                                                 a_column_name="sentence_a",
                                                                                 b_column_name="candidates",
                                                                                 candidate_num=self.train_candidate_num,
                                                                                 suffix=save_load_suffix),)

            # read val data. exist? read!--------------------------------------------------
            if pair_flag:
                process_val_test_func = self.__tokenize_match_multi_candidate_data_then_save
            else:
                process_val_test_func = self.__tokenize_match_cross_data_then_save

            if not get_val:
                pass
            elif os.path.exists(root_path + "dataset/" + save_load_prefix + self.dataset_name + "_val"):
                val_datasets = (torch.load(root_path + "dataset/" + save_load_prefix + self.dataset_name + "_val")['dataset'],)
            else:
                string_val_dataset = datasets.load_from_disk(root_path + "dataset/string_dev_" + self.dataset_name)

                val_datasets = (process_val_test_func(data=string_val_dataset,
                                                      save_name=save_load_prefix + self.dataset_name + "_val",
                                                      a_column_name="sentence_a",
                                                      b_column_name="candidates",
                                                      candidate_num=self.val_candidate_num),)

            # read test data. exist? read!--------------------------------------------------
            if not get_test:
                pass
            elif os.path.exists(root_path + "dataset/" + save_load_prefix + self.dataset_name + "_test"):
                test_datasets = (torch.load(root_path + "dataset/" + save_load_prefix + self.dataset_name + "_test")['dataset'],)
            else:
                string_test_dataset = datasets.load_from_disk(root_path + "dataset/string_test_" + self.dataset_name)

                test_datasets = (process_val_test_func(data=string_test_dataset,
                                                      save_name=save_load_prefix + self.dataset_name + "_test",
                                                      a_column_name="sentence_a",
                                                      b_column_name="candidates",
                                                      candidate_num=self.val_candidate_num),)
        else:
            raise_dataset_error()

        return train_datasets, val_datasets, test_datasets

    def __create_model(self):
        print("---------------------- create model ----------------------")
        # 创建自己的model
        # add model
        if self.model_class in ['QAClassifierModel']:
            model = QAClassifierModel(config=self.config)
        elif self.model_class in ['CrossBERT']:
            model = CrossBERT(config=self.config)
        elif self.model_class in ['ClassifyParallelEncoder']:
            model = ClassifyParallelEncoder(config=self.config)
        elif self.model_class in ['CLSClassifyParallelEncoder']:
            model = CLSClassifyParallelEncoder(config=self.config)
        elif self.model_class in ['DisenClassifyParallelEncoder']:
            model = DisenClassifyParallelEncoder(config=self.config)
        elif self.model_class in ['DisenCLSClassifyParallelEncoder']:
            model = DisenCLSClassifyParallelEncoder(config=self.config)
        elif self.model_class in ['PolyEncoder']:
            model = PolyEncoder(config=self.config)
        elif self.model_class in ['QAMatchModel']:
            model = QAMatchModel(config=self.config)
        elif self.model_class in ['CMCModel']:
            model = CMCModel(config=self.config)
        elif self.model_class in ['ColBERT']:
            model = ColBERT(config=self.config)
        elif self.model_class in ['MatchParallelEncoder']:
            model = MatchParallelEncoder(config=self.config)
        elif self.model_class == 'CLSMatchParallelEncoder':
            model = CLSMatchParallelEncoder(config=self.config)
        elif self.model_class == 'DisenMatchParallelEncoder':
            model = DisenMatchParallelEncoder(config=self.config)
        elif self.model_class == 'DisenCLSMatchParallelEncoder':
            model = DisenCLSMatchParallelEncoder(config=self.config)
        elif self.model_class in ['ClassifyDeformer']:
            model = ClassifyDeformer(config=self.config)
        elif self.model_class in ['MatchDeformer']:
            model = MatchDeformer(config=self.config)
        elif self.model_class in ['MatchCrossBERT']:
            model = MatchCrossBERT(config=self.config)
        else:
            raise Exception("This model class is not supported for creating!!")

        # 要不要加载现成的模型
        if self.load_model_flag:
            model = self.load_model(model, self.load_model_path)

        print("--------------------- model  created ---------------------")

        return model

    # 读取命令行传入的参数
    def __read_args_for_train(self, args):
        self.evaluate_epoch = args.evaluate_epoch
        self.clean_graph = args.clean_graph
        self.in_batch = args.in_batch
        self.top_layer_num = args.top_layer_num
        self.no_apex = args.no_apex
        self.val_batch_size = args.val_batch_size
        self.text_max_len = args.text_max_len
        self.dataset_name = args.dataset_name
        self.model_class = args.model_class
        self.train_candidate_num = args.train_candidate_num
        self.load_model_flag = args.load_model
        self.dataset_split_num = args.dataset_split_num
        self.train_batch_size = args.train_batch_size
        self.model_save_prefix = args.model_save_prefix
        self.no_aggregator = args.no_aggregator
        self.no_enricher = args.no_enricher

        self.local_rank = args.local_rank
        self.data_parallel = args.data_parallel
        self.data_distribute = args.data_distribute
        self.distill_flag = args.distill
        self.teacher_path = args.teacher_path

        self.num_train_epochs = args.num_train_epochs
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.no_initial_test = args.no_initial_test
        self.first_stage_lr = args.first_stage_lr

        self.composition = args.composition
        self.restore_flag = args.restore
        self.label_num = args.label_num
        self.load_model_path = args.load_model_path
        self.save_model_dict = args.save_model_dict
        self.last_model_dict = args.last_model_dict
        self.context_num = args.context_num
        self.query_block_size = args.query_block_size
        self.first_seq_max_len = args.first_seq_max_len
        self.step_max_query_num = args.step_max_query_num

        # add dataset
        if self.dataset_name == 'ubuntu':
            self.val_candidate_num = 10
            self.r_k_num = (1,)
        elif self.dataset_name == 'dstc7':
            self.val_candidate_num = 100
            self.r_k_num = (1, 10)
        elif self.dataset_name == 'yahooqa':
            self.val_candidate_num = 5
            self.r_k_num = (1, )
        elif self.dataset_name in ['msmarco', 'hard_msmarco']:
            self.val_candidate_num = 1000
            self.r_k_num = (1,10)
        elif self.dataset_name in ['mnli', 'qqp', 'boolq']:
            self.val_candidate_num = None
            self.r_k_num = None
        else:
            raise_dataset_error()

    def __match_string_list(self, this_string, string_list):
        for s in string_list:
            if s in this_string:
                return True
        
        return False
    
    # 读取命令行传入的有关config的参数
    def __read_args_for_config(self, args):
        if self.__match_string_list(args.pretrained_bert_path, ['prajjwal1/bert-small', 'google/bert_uncased_L-6_H-512_A-8', 'google/bert_uncased_L-8_H-512_A-8', 'prajjwal1/bert-medium']):
            word_embedding_len = 512
            sentence_embedding_len = 512
        elif self.__match_string_list(args.pretrained_bert_path, ['bert-base-uncased', 'Luyu/co-condenser-marco-retriever']):
            word_embedding_len = 768
            sentence_embedding_len = 768
        elif self.__match_string_list(args.pretrained_bert_path, ['google/bert_uncased_L-2_H-128_A-2']):
            word_embedding_len = 128
            sentence_embedding_len = 128
        else:
            raise Exception("word_embedding_len, sentence_embedding_len is needed!")

        # add model
        if self.model_class in ['QAClassifierModel', 'QAMatchModel', 'CMCModel']:
            config = QAClassifierModelConfig(len(self.tokenizer),
                                         pretrained_bert_path=args.pretrained_bert_path,
                                         num_labels=args.label_num,
                                         word_embedding_len=word_embedding_len,
                                         sentence_embedding_len=sentence_embedding_len,
                                         composition=self.composition,
                                         teacher_loss=args.teacher_loss,
                                         alpha=args.alpha)
        elif self.model_class in ['ColBERT']:
            config = ColBERTConfig(pretrained_bert_path=args.pretrained_bert_path,
                                   word_embedding_len=word_embedding_len,
                                   sentence_embedding_len=128)
        elif self.model_class in ['CrossBERT', 'MatchCrossBERT']:
            config = CrossBERTConfig(len(self.tokenizer),
                                     pretrained_bert_path=args.pretrained_bert_path,
                                     num_labels=args.label_num,
                                     word_embedding_len=word_embedding_len,
                                     sentence_embedding_len=sentence_embedding_len,
                                     composition=self.composition)
        elif self.model_class in ['ClassifyParallelEncoder', 'MatchParallelEncoder',
                                  'CLSMatchParallelEncoder', 'CLSClassifyParallelEncoder',
                                  'DisenMatchParallelEncoder', 'DisenCLSMatchParallelEncoder',
                                  'DisenCLSClassifyParallelEncoder', 'DisenClassifyParallelEncoder']:
            config = ParallelEncoderConfig(len(self.tokenizer),
                                           pretrained_bert_path=args.pretrained_bert_path,
                                           num_labels=args.label_num,
                                           word_embedding_len=word_embedding_len,
                                           sentence_embedding_len=sentence_embedding_len,
                                           composition=self.composition,
                                           context_num=self.context_num,
                                           text_max_len=self.text_max_len,
                                           used_layers=args.used_layers)
        elif self.model_class == 'PolyEncoder':
            config = PolyEncoderConfig(len(self.tokenizer),
                                       pretrained_bert_path=args.pretrained_bert_path,
                                       num_labels=args.label_num,
                                       word_embedding_len=word_embedding_len,
                                       sentence_embedding_len=sentence_embedding_len,
                                       context_num=self.context_num)
        elif self.model_class in ['ClassifyDeformer', 'MatchDeformer']:
            config = DeformerConfig(len(self.tokenizer),
                                    pretrained_bert_path=args.pretrained_bert_path,
                                    num_labels=args.label_num,
                                    word_embedding_len=word_embedding_len,
                                    sentence_embedding_len=sentence_embedding_len,
                                    top_layer_num=self.top_layer_num)
        else:
            raise Exception("No config for this class!")

        return config

    # 根据model_class获取optimizer
    def __get_optimizer(self, final_stage_flag):

        model = self.model.module if hasattr(self.model, 'module') else self.model

        # add model
        # 获得模型的训练参数和对应的学习率
        if self.model_class in ['QAClassifierModel']:
            parameters_dict_list = [
                # 这几个一样
                {'params': model.bert_model.parameters(), 'lr': 1e-5},
                # 这几个一样
                {'params': model.self_attention_weight_layer.parameters(), 'lr': 1e-5},
                {'params': model.value_layer.parameters(), 'lr': 1e-5},
                # 这个不设定
                {'params': model.classifier.parameters(), 'lr': 1e-5}
            ]
        elif self.model_class in ['CrossBERT', 'MatchCrossBERT']:
            parameters_dict_list = [
                # 这几个一样
                {'params': model.bert_model.parameters(), 'lr': 1e-5},
            ]
        elif self.model_class in ['ClassifyParallelEncoder', 'DisenClassifyParallelEncoder']:
            parameters_dict_list = [
                # 这几个一样
                {'params': model.bert_model.parameters(), 'lr': 1e-5},
                {'params': model.composition_layer.parameters(), 'lr': 1e-5},
                {'params': model.decoder.parameters(), 'lr': 1e-5},
                {'params': model.classifier.parameters(), 'lr': 1e-5},
            ]
        elif self.model_class in ['CLSClassifyParallelEncoder', 'DisenCLSClassifyParallelEncoder']:
            parameters_dict_list = [
                # 这几个一样
                {'params': model.bert_model.parameters(), 'lr': 1e-5},
                {'params': model.context_cls, 'lr': 1e-5},
                {'params': model.decoder.parameters(), 'lr': 1e-5},
                {'params': model.classifier.parameters(), 'lr': 1e-5},
            ]
        elif self.model_class in ['MatchParallelEncoder', 'DisenMatchParallelEncoder']:
            parameters_dict_list = [
                # 这几个一样
                {'params': model.bert_model.parameters(), 'lr': 1e-5},
                {'params': model.composition_layer.parameters(), 'lr': 1e-5},
                {'params': model.decoder.parameters(), 'lr': 1e-5},
            ]
        elif self.model_class in ['CLSMatchParallelEncoder', 'DisenCLSMatchParallelEncoder']:
            parameters_dict_list = [
                # 这几个一样
                {'params': model.bert_model.parameters(), 'lr': 1e-5},
                {'params': model.context_cls, 'lr': 1e-5},
                {'params': model.decoder.parameters(), 'lr': 1e-5},
                # {'params': model.classifier.parameters(), 'lr': 1e-5},
            ]
        elif self.model_class == 'PolyEncoder':
            parameters_dict_list = [
                # 这几个一样
                {'params': model.bert_model.parameters(), 'lr': 1e-5},
                {'params': model.query_composition_layer.parameters(), 'lr': 1e-5},
                {'params': model.classifier.parameters(), 'lr': 1e-5},
            ]
        elif self.model_class == 'ColBERT':
            parameters_dict_list = [
                # 这几个一样
                {'params': model.bert.parameters(), 'lr': 1e-5},
                {'params': model.linear.parameters(), 'lr': 1e-5},
            ]
        elif self.model_class == 'QAMatchModel':
            parameters_dict_list = [
                # 这几个一样
                {'params': model.bert_model.parameters(), 'lr': 1e-5},
            ]
        elif self.model_class == 'CMCModel':
            parameters_dict_list = [
                {'params': model.qry_bert_model.parameters(), 'lr': 1e-5},
                {'params': model.can_bert_model.parameters(), 'lr': 1e-5},
                {'params': model.extend_multi_transformerencoder.parameters(), 'lr': 1e-5},
                {'params': model.extend_multi_transformerencoderlayer.parameters(), 'lr': 1e-5},
            ]
            
        elif self.model_class in ['ClassifyDeformer', 'MatchDeformer']:
            parameters_dict_list = [
                # 这几个一样
                {'params': model.bert_model.parameters(), 'lr': 1e-5},
            ]
        else:
            raise Exception("No optimizer supported for this model class!")

        # 对于那些有两段的,第一段训练参数不太一样
        if not final_stage_flag:
            if self.model_class in ['ClassifyParallelEncoder', 'DisenClassifyParallelEncoder']:
                parameters_dict_list = [
                    # 这几个一样
                    {'params': model.decoder.parameters(), 'lr': 1e-5},
                    {'params': model.composition_layer.parameters(), 'lr': 1e-5},
                    {'params': model.classifier.parameters(), 'lr': 1e-5},
                ]
            elif self.model_class in ['CLSClassifyParallelEncoder', 'DisenCLSClassifyParallelEncoder']:
                parameters_dict_list = [
                    # 这几个一样
                    {'params': model.decoder.parameters(), 'lr': 1e-5},
                    {'params': model.context_cls, 'lr': 1e-5},
                    {'params': model.classifier.parameters(), 'lr': 1e-5},
                ]
            elif self.model_class in ['MatchParallelEncoder', 'DisenMatchParallelEncoder']:
                parameters_dict_list = [
                    # 这几个一样
                    {'params': model.composition_layer.parameters(), 'lr': 1e-5},
                    {'params': model.decoder.parameters(), 'lr': 1e-5},
                ]
            elif self.model_class in ['CLSMatchParallelEncoder', 'DisenCLSMatchParallelEncoder']:
                parameters_dict_list = [
                    # 这几个一样
                    {'params': model.context_cls, 'lr': 1e-5},
                    {'params': model.decoder.parameters(), 'lr': 1e-5},
                ]
            else:
                raise Exception("Have Two Stage But No optimizer supported for this model class!")

        # if to restore, it will be printed in other places
        if not self.restore_flag:
            print(parameters_dict_list)
        optimizer = torch.optim.AdamW(parameters_dict_list, lr=1e-5) # jcy 원래 5e-5(4581), 4583는 1e-5로 돌리고 command line first stage lr도 0.00001로 해두고 context_num 64로
        print("*"*30)

        return optimizer

    def __model_to_device(self):
        self.model.to(self.device)
        self.model.train()

    def __model_parallel(self):
        if self.data_distribute:
            # self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            # self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
            #                                                        find_unused_parameters=True,
            #                                                        output_device=self.local_rank)
            self.model = convert_syncbn_model(self.model)
            self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank])
            # self.model = convert_syncbn_model(self.model).to(self.device)
            # self.model, new_optimizer = amp.initialize(self.model, optimizer, opt_level='O1')
            # self.model = DistributedDataParallel(self.model, delay_allreduce=True)

        elif self.data_parallel:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)

    # 输入为QA，而非title.body.answer的模型的训练步
    def __classify_train_step_for_qa_input(self, batch, optimizer, now_batch_num, scheduler, **kwargs):
        cross_entropy_function = nn.CrossEntropyLoss()

        # 读取数据
        a_input_ids = (batch['a_input_ids']).to(self.device)
        a_token_type_ids = (batch['a_token_type_ids']).to(self.device)
        a_attention_mask = (batch['a_attention_mask']).to(self.device)

        b_input_ids = (batch['b_input_ids']).to(self.device)
        b_token_type_ids = (batch['b_token_type_ids']).to(self.device)
        b_attention_mask = (batch['b_attention_mask']).to(self.device)

        qa_labels = (batch['label']).to(self.device)

        # 得到模型的结果
        logits = self.model(
            a_input_ids=a_input_ids, a_token_type_ids=a_token_type_ids,
            a_attention_mask=a_attention_mask,
            b_input_ids=b_input_ids, b_token_type_ids=b_token_type_ids,
            b_attention_mask=b_attention_mask, no_aggregator=self.no_aggregator, no_enricher=self.no_enricher)

        # 计算损失
        step_loss = cross_entropy_function(logits, qa_labels)

        # 误差反向传播
        if not APEX_FLAG or self.no_apex:
            step_loss.backward()
        else:
            with amp.scale_loss(step_loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        # 更新模型参数
        if (now_batch_num + 1) % self.gradient_accumulation_steps == 0:
            # add model
            # only update memory in embeddings
            if self.model_class in ['ClassifyParallelEncoder', 'CLSClassifyParallelEncoder',
                                    'DisenCLSClassifyParallelEncoder', 'DisenClassifyParallelEncoder']:
                nn.utils.clip_grad_norm_(self.model.decoder['LSTM'].parameters(), max_norm=20, norm_type=2)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # 统计命中率
        step_shoot_num = logits.shape[0]
        with torch.no_grad():
            _, row_max_indices = logits.topk(k=1, dim=-1)
            step_hit_num = 0
            for i, max_index in enumerate(row_max_indices):
                inner_index = max_index[0]
                if inner_index == qa_labels[i]:
                    step_hit_num += 1

        return (step_loss, step_shoot_num, step_hit_num)

    # 输入为一个q和很多歌candidate，最佳答案放在最后一个。
    def __train_step_for_multi_candidates_input(self, batch, optimizer, now_batch_num, scheduler, **kwargs):
        cross_entropy_function = nn.CrossEntropyLoss()
        # 读取数据
        a_input_ids = (batch['a_input_ids']).to(self.device)
        a_token_type_ids = (batch['a_token_type_ids']).to(self.device)
        a_attention_mask = (batch['a_attention_mask']).to(self.device)

        b_input_ids = (batch['b_input_ids']).to(self.device)
        b_token_type_ids = (batch['b_token_type_ids']).to(self.device)
        b_attention_mask = (batch['b_attention_mask']).to(self.device)

        candidate_num = b_input_ids.shape[1]

        in_batch_negative_strategy = False
        # 得到模型的结果
        logits = self.model(
            a_input_ids=a_input_ids, a_token_type_ids=a_token_type_ids,
            a_attention_mask=a_attention_mask,
            b_input_ids=b_input_ids, b_token_type_ids=b_token_type_ids,
            b_attention_mask=b_attention_mask, train_flag=in_batch_negative_strategy,
            return_dot_product=in_batch_negative_strategy, no_aggregator=self.no_aggregator, no_enricher=self.no_enricher)

        if in_batch_negative_strategy:
            qa_labels = []
            for i in range(logits.shape[0]):
                qa_labels.append(i*candidate_num + candidate_num - 1)
        else:
            # best candidate at last
            qa_labels = [candidate_num - 1] * logits.shape[0]

        qa_labels = torch.tensor(qa_labels, dtype=torch.long, device=logits.device)

        # 计算损失
        step_loss = cross_entropy_function(logits, qa_labels)

        # 误差反向传播
        if not APEX_FLAG or self.no_apex:
            step_loss.backward()
        else:
            with amp.scale_loss(step_loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        # 更新模型参数
        if (now_batch_num + 1) % self.gradient_accumulation_steps == 0:
            if self.model_class in ['MatchParallelEncoder', 'CLSMatchParallelEncoder',
                                    'DisenMatchParallelEncoder', 'DisenCLSMatchParallelEncoder']:
                # for p in self.model.decoder['LSTM'].parameters():
                #     print(p.grad.norm)
                # print("*******************************"*3)
                # for p in self.model.decoder.layer_chunks.parameters():
                #     print(p.grad.norm)
                # raise_test_error()

                nn.utils.clip_grad_norm_(self.model.decoder['LSTM'].parameters(), max_norm=20, norm_type=2)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # 统计命中率
        step_shoot_num = logits.shape[0]

        _, row_max_indices = logits.topk(k=1, dim=-1)
        step_hit_num = (row_max_indices.squeeze(-1) == qa_labels).sum().item()

        # step_shoot_num = a_input_ids.shape[0]
        # step_hit_num = a_input_ids.shape[0]
        return (step_loss, step_shoot_num, step_hit_num)

    # 输入为QA，而非title.body.answer的模型的训练步
    def __match_train_step_for_qa_input(self, batch, optimizer, now_batch_num, scheduler, **kwargs):
        # add model
        # 读取数据
        a_input_ids = (batch['a_input_ids']).to(self.device)
        a_token_type_ids = (batch['a_token_type_ids']).to(self.device)
        a_attention_mask = (batch['a_attention_mask']).to(self.device)

        b_input_ids = (batch['b_input_ids']).to(self.device)
        b_token_type_ids = (batch['b_token_type_ids']).to(self.device)
        b_attention_mask = (batch['b_attention_mask']).to(self.device)

        step_loss = self.model(
            a_input_ids=a_input_ids, a_token_type_ids=a_token_type_ids,
            a_attention_mask=a_attention_mask,
            b_input_ids=b_input_ids, b_token_type_ids=b_token_type_ids,
            b_attention_mask=b_attention_mask, train_flag=True, match_train=True,
            no_aggregator=self.no_aggregator, no_enricher=self.no_enricher,
            no_apex=self.no_apex,
            optimizer=optimizer,
            is_hard_msmarco=(self.dataset_name == 'hard_msmarco'),
        )

        # 误差反向传播
        if not APEX_FLAG or self.no_apex:
            step_loss.backward()
        else:
            with amp.scale_loss(step_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        #
        # if self.model_class in ['MatchParallelEncoder', 'CLSMatchParallelEncoder',
                                # 'DisenMatchParallelEncoder', 'DisenCLSMatchParallelEncoder']:
        #     nn.utils.clip_grad_norm_(self.model.decoder['LSTM'].parameters(), max_norm=20, norm_type=2)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        return (step_loss,)

    # 支持先计算candidate的表征，接着按块进行query和candidate的匹配 # 주석 : candidate의 representation을 먼저 계산하고, 그 다음 query와 candidate를 블록 단위로 일치시킴
    def __efficient_match_train_step_for_qa_input(self, batch, optimizer, now_batch_num, scheduler, **kwargs):
        model = self.model.module if hasattr(self.model, 'module') else self.model

        b_input_ids = (batch['b_input_ids']).to(self.device)
        b_token_type_ids = (batch['b_token_type_ids']).to(self.device)
        b_attention_mask = (batch['b_attention_mask']).to(self.device)

        if self.model_class == 'MatchDeformer':
            # (batch_size, seq_len, dim)
            candidate_embeddings, clean_b_attention_mask = model.prepare_candidates(input_ids=b_input_ids,
                                                                                    token_type_ids=b_token_type_ids,
                                                                                    attention_mask=b_attention_mask)
        else:
            # (batch_size, (context_num / seq len), dim)
            candidate_embeddings = model.prepare_candidates(input_ids=b_input_ids,
                                                            token_type_ids=b_token_type_ids,
                                                            attention_mask=b_attention_mask)

        # check validity
        if self.step_max_query_num == -1:
            raise Exception(
                "Please designate arg:step_max_query_num according to cuda. Must be smaller than batch size.")

        # 读取数据
        a_input_ids = (batch['a_input_ids'])
        a_token_type_ids = (batch['a_token_type_ids'])
        a_attention_mask = (batch['a_attention_mask'])

        batch_size = a_input_ids.shape[0]

        # begin training
        batch_count = 0
        whole_dot_product = []
        while batch_count < batch_size:
            new_batch_count = min(batch_count + self.step_max_query_num, batch_size)

            # add model
            if self.model_class in ['MatchParallelEncoder', 'CLSMatchParallelEncoder',
                                    'DisenMatchParallelEncoder', 'DisenCLSMatchParallelEncoder']:
                this_candidate_embeddings = candidate_embeddings.unsqueeze(0).expand(new_batch_count-batch_count, -1, -1, -1)
            else:
                this_candidate_embeddings = candidate_embeddings

            this_a_input_ids = a_input_ids[batch_count:new_batch_count].to(self.device)
            this_a_token_type_ids = a_token_type_ids[batch_count:new_batch_count].to(self.device)
            this_a_attention_mask = a_attention_mask[batch_count:new_batch_count].to(self.device)

            if self.model_class == 'MatchDeformer':
                a_embeddings, clean_a_attention_mask = model.prepare_candidates(input_ids=this_a_input_ids,
                                                                                token_type_ids=this_a_token_type_ids,
                                                                                attention_mask=this_a_attention_mask,
                                                                                candidate_flag=False)
                # print(a_embeddings.shape, candidate_embeddings.shape)
                joint_embeddings = model.joint_encoding(a_embeddings=a_embeddings,
                                                        b_embeddings=candidate_embeddings,
                                                        a_attention_mask=clean_a_attention_mask,
                                                        b_attention_mask=clean_b_attention_mask,
                                                        train_flag=True)
                pooler = model.pooler_layer(joint_embeddings)
                dot_product = model.classifier(pooler).reshape(-1, batch_size)
                # print(dot_product.shape)
                # raise_test_error()
            else:
                dot_product = model.do_queries_match(input_ids=this_a_input_ids,
                                                     token_type_ids=this_a_token_type_ids,
                                                     attention_mask=this_a_attention_mask,
                                                     candidate_context_embeddings=this_candidate_embeddings,
                                                     no_aggregator=self.no_aggregator,
                                                     no_enricher=self.no_enricher,
                                                     train_flag=True)

            batch_count = new_batch_count
            whole_dot_product.append(dot_product)

        whole_dot_product = torch.cat(whole_dot_product, dim=0)
        mask = torch.eye(whole_dot_product.size(0)).to(whole_dot_product.device)
        # here can be modified as:
        # loss = F.log_softmax(whole_dot_product, dim=-1) * mask + F.log_softmax(whole_dot_product, dim=0) * mask
        loss = F.log_softmax(whole_dot_product, dim=-1) * mask
        step_loss = (-loss.sum(dim=1)).mean()

        # 误差反向传播
        if not APEX_FLAG or self.no_apex:
            step_loss.backward()
        else:
            with amp.scale_loss(step_loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        if self.model_class in ['MatchParallelEncoder', 'CLSMatchParallelEncoder',
                                'DisenMatchParallelEncoder', 'DisenCLSMatchParallelEncoder']:
            nn.utils.clip_grad_norm_(self.model.decoder['LSTM'].parameters(), max_norm=20, norm_type=2)

        # 更新模型参数
        if (now_batch_num + 1) % self.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        return (step_loss,)

    # 多卡，并且按块进行query和candidate的匹配，每一块进行一次后向传播，用以减少计算图的显存占用,但这会增加运算时间.
    def __data_parallel_match_train_step_for_qa_input(self, batch, optimizer, now_batch_num, scheduler, **kwargs):
        model = self.model.module if hasattr(self.model, 'module') else self.model

        batch_size = batch['b_input_ids'].shape[0]

        # begin training
        batch_count = 0
        all_loss = torch.tensor(0.0, device=self.device)

        while batch_count < batch_size:
            b_input_ids = (batch['b_input_ids']).to(self.device)
            b_token_type_ids = (batch['b_token_type_ids']).to(self.device)
            b_attention_mask = (batch['b_attention_mask']).to(self.device)

            # (batch_size, (context_num), dim)
            candidate_context_embeddings = self.model(a_input_ids=b_input_ids,
                                                      a_token_type_ids=b_token_type_ids,
                                                      a_attention_mask=b_attention_mask,
                                                      b_input_ids=None,
                                                      b_token_type_ids=None,
                                                      b_attention_mask=None,
                                                      train_flag=True,
                                                      prepare_candidates=True)

            # check validity
            if self.step_max_query_num == -1:
                raise Exception(
                    "Please designate arg:step_max_query_num for __efficient_match_train_step_for_qa_input.")

            new_batch_count = min(batch_size, batch_count + self.step_max_query_num)

            # add model
            if self.model_class in ['MatchParallelEncoder', 'CLSMatchParallelEncoder',
                                    'DisenMatchParallelEncoder', 'DisenCLSMatchParallelEncoder']:
                candidate_context_embeddings = candidate_context_embeddings.unsqueeze_(0).expand(
                    new_batch_count - batch_count, -1, -1,
                    -1)

            this_a_input_ids = batch['a_input_ids'][batch_count:new_batch_count].to(self.device)
            this_a_token_type_ids = batch['a_token_type_ids'][batch_count:new_batch_count].to(self.device)
            this_a_attention_mask = batch['a_attention_mask'][batch_count:new_batch_count].to(self.device)

            dot_product = self.model(a_input_ids=this_a_input_ids,
                                     a_token_type_ids=this_a_token_type_ids,
                                     a_attention_mask=this_a_attention_mask,
                                     b_input_ids=None,
                                     b_token_type_ids=None,
                                     b_attention_mask=None,
                                     train_flag=True,
                                     do_queries_match=True,
                                     candidate_context_embeddings=candidate_context_embeddings,
                                     no_aggregator=self.no_aggregator,
                                     no_enricher=self.no_enricher)

            # calculate gradient
            mask = (torch.eye(batch_size).to(self.device))[batch_count:new_batch_count]
            loss = F.log_softmax(dot_product, dim=-1) * mask
            step_loss = ((-loss.sum(dim=1)).sum())/batch_size
            all_loss += step_loss

            # 误差反向传播
            if not APEX_FLAG or self.no_apex:
                step_loss.backward()
            else:
                with amp.scale_loss(step_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

            batch_count = new_batch_count

        if self.model_class in ['MatchParallelEncoder', 'CLSMatchParallelEncoder',
                                'DisenMatchParallelEncoder', 'DisenCLSMatchParallelEncoder']:
            nn.utils.clip_grad_norm_(model.decoder['LSTM'].parameters(), max_norm=20, norm_type=2)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        return (all_loss,)

    # 按块进行query和candidate的匹配，并且每一块进行一次后向传播，用以减少计算图的显存占用,但这会增加运算时间
    def __clean_graph_efficient_match_train_step_for_bi_hard_msmarco(self, batch, optimizer, now_batch_num, scheduler,
                                                                     **kwargs):
        loss_func = torch.nn.CrossEntropyLoss(reduction='sum')

        model = self.model.module if hasattr(self.model, 'module') else self.model

        batch_size = batch['a_input_ids'].shape[0]

        # begin training
        batch_count = 0
        all_loss = torch.tensor(0.0, device=self.device)

        while batch_count < batch_size:
            # positive
            b_input_ids = (batch['b_input_ids']).to(self.device)
            b_token_type_ids = (batch['b_token_type_ids']).to(self.device)
            b_attention_mask = (batch['b_attention_mask']).to(self.device)

            # hard negative
            c_input_ids = (batch['c_input_ids']).to(self.device)
            c_token_type_ids = (batch['c_token_type_ids']).to(self.device)
            c_attention_mask = (batch['c_attention_mask']).to(self.device)

            candidate_input_ids = torch.cat([b_input_ids, c_input_ids], dim=0)
            candidate_token_type_ids = torch.cat([b_token_type_ids, c_token_type_ids], dim=0)
            candidate_attention_mask = torch.cat([b_attention_mask, c_attention_mask], dim=0)

            # (batch_size, (context_num), dim)
            candidate_embeddings = model.prepare_candidates(input_ids=candidate_input_ids,
                                                            token_type_ids=candidate_token_type_ids,
                                                            attention_mask=candidate_attention_mask)

            # check validity
            if self.step_max_query_num == -1:
                raise Exception(
                    "Please designate arg:step_max_query_num for __efficient_match_train_step_for_qa_input.")

            # 接下来进行匹配
            new_batch_count = min(batch_count + self.step_max_query_num, batch_size)

            # add model
            if self.model_class in ['MatchParallelEncoder', 'CLSMatchParallelEncoder',
                                    'DisenMatchParallelEncoder', 'DisenCLSMatchParallelEncoder']:
                candidate_embeddings = candidate_embeddings.unsqueeze_(0).expand(new_batch_count - batch_count, -1,
                                                                                 -1, -1)

            this_a_input_ids = batch['a_input_ids'][batch_count:new_batch_count].to(self.device)
            this_a_token_type_ids = batch['a_token_type_ids'][batch_count:new_batch_count].to(self.device)
            this_a_attention_mask = batch['a_attention_mask'][batch_count:new_batch_count].to(self.device)

            dot_product = model.do_queries_match(input_ids=this_a_input_ids,
                                                 token_type_ids=this_a_token_type_ids,
                                                 attention_mask=this_a_attention_mask,
                                                 candidate_context_embeddings=candidate_embeddings,
                                                 no_aggregator=self.no_aggregator,
                                                 no_enricher=self.no_enricher,
                                                 train_flag=True)

            # calculate gradient
            labels = torch.tensor(range(batch_size), dtype=torch.long, device=self.device)[batch_count:new_batch_count]

            loss = loss_func(dot_product, labels)
            step_loss = loss / batch_size
            all_loss += step_loss

            # 误差反向传播
            if not APEX_FLAG or self.no_apex:
                step_loss.backward()
            else:
                with amp.scale_loss(step_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

            batch_count = new_batch_count

        if self.model_class in ['MatchParallelEncoder', 'CLSMatchParallelEncoder',
                                'DisenMatchParallelEncoder', 'DisenCLSMatchParallelEncoder']:
            nn.utils.clip_grad_norm_(self.model.decoder['LSTM'].parameters(), max_norm=20, norm_type=2)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        return (all_loss,)

    # 按块进行query和candidate的匹配，并且每一块进行一次后向传播，用以减少计算图的显存占用,但这会增加运算时间
    def __clean_graph_efficient_match_train_step_for_qa_input(self, batch, optimizer, now_batch_num, scheduler, **kwargs):
        model = self.model.module if hasattr(self.model, 'module') else self.model

        batch_size = batch['b_input_ids'].shape[0]

        # begin training
        batch_count = 0
        all_loss = torch.tensor(0.0, device=self.device)

        while batch_count < batch_size:
            b_input_ids = (batch['b_input_ids']).to(self.device)
            b_token_type_ids = (batch['b_token_type_ids']).to(self.device)
            b_attention_mask = (batch['b_attention_mask']).to(self.device)

            if self.model_class == 'MatchDeformer':
                # (batch_size, seq_len, dim)
                candidate_embeddings, clean_b_attention_mask = model.prepare_candidates(input_ids=b_input_ids,
                                                                                        token_type_ids=b_token_type_ids,
                                                                                        attention_mask=b_attention_mask)
            else:
                # (batch_size, (context_num / seq len), dim)
                candidate_embeddings = model.prepare_candidates(input_ids=b_input_ids,
                                                                token_type_ids=b_token_type_ids,
                                                                attention_mask=b_attention_mask)

            # check validity
            if self.step_max_query_num == -1:
                raise Exception(
                    "Please designate arg:step_max_query_num for __efficient_match_train_step_for_qa_input.")

            # 接下来进行匹配
            new_batch_count = min(batch_count + self.step_max_query_num, batch_size)

            # add model
            if self.model_class in ['MatchParallelEncoder', 'CLSMatchParallelEncoder',
                                    'DisenMatchParallelEncoder', 'DisenCLSMatchParallelEncoder']:
                candidate_embeddings = candidate_embeddings.unsqueeze_(0).expand(new_batch_count-batch_count, -1, -1, -1)

            this_a_input_ids = batch['a_input_ids'][batch_count:new_batch_count].to(self.device)
            this_a_token_type_ids = batch['a_token_type_ids'][batch_count:new_batch_count].to(self.device)
            this_a_attention_mask = batch['a_attention_mask'][batch_count:new_batch_count].to(self.device)

            if self.model_class == 'MatchDeformer':
                a_embeddings, clean_a_attention_mask = model.prepare_candidates(input_ids=this_a_input_ids,
                                                                                token_type_ids=this_a_token_type_ids,
                                                                                attention_mask=this_a_attention_mask,
                                                                                candidate_flag=False)
                # print(a_embeddings.shape, candidate_embeddings.shape)
                joint_embeddings = model.joint_encoding(a_embeddings=a_embeddings,
                                                        b_embeddings=candidate_embeddings,
                                                        a_attention_mask=clean_a_attention_mask,
                                                        b_attention_mask=clean_b_attention_mask,
                                                        train_flag=True)
                pooler = model.pooler_layer(joint_embeddings)
                dot_product = model.classifier(pooler).reshape(-1, batch_size)
                # print(dot_product.shape)
                # raise_test_error()
            else:
                dot_product = model.do_queries_match(input_ids=this_a_input_ids,
                                                     token_type_ids=this_a_token_type_ids,
                                                     attention_mask=this_a_attention_mask,
                                                     candidate_context_embeddings=candidate_embeddings,
                                                     no_aggregator=self.no_aggregator,
                                                     no_enricher=self.no_enricher,
                                                     train_flag=True)

            # calculate gradient
            mask = (torch.eye(batch_size).to(self.device))[batch_count:new_batch_count]
            loss = F.log_softmax(dot_product, dim=-1) * mask
            step_loss = ((-loss.sum(dim=1)).sum())/batch_size
            all_loss += step_loss

            # 误差反向传播
            if not APEX_FLAG or self.no_apex:
                step_loss.backward()
            else:
                with amp.scale_loss(step_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

            batch_count = new_batch_count

        if self.model_class in ['MatchParallelEncoder', 'CLSMatchParallelEncoder',
                                'DisenMatchParallelEncoder', 'DisenCLSMatchParallelEncoder']:
            nn.utils.clip_grad_norm_(self.model.decoder['LSTM'].parameters(), max_norm=20, norm_type=2)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        return (all_loss,)

    # cross模型的训练步
    def __classify_train_step_for_cross(self, batch, optimizer, now_batch_num, scheduler, **kwargs):
        cross_entropy_function = nn.CrossEntropyLoss()

        # 读取数据
        input_ids = (batch['input_ids']).to(self.device)
        token_type_ids = (batch['token_type_ids']).to(self.device)
        attention_mask = (batch['attention_mask']).to(self.device)

        qa_labels = (batch['label']).to(self.device)

        # 得到模型的结果
        logits = self.model(
            input_ids=input_ids, token_type_ids=token_type_ids,
            attention_mask=attention_mask)

        # 计算损失
        step_loss = cross_entropy_function(logits, qa_labels)

        # 误差反向传播
        if not APEX_FLAG or self.no_apex:
            step_loss.backward()
        else:
            with amp.scale_loss(step_loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        # 更新模型参数
        if (now_batch_num + 1) % self.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # 统计命中率
        step_shoot_num = logits.shape[0]
        with torch.no_grad():
            _, row_max_indices = logits.topk(k=1, dim=-1)
            step_hit_num = 0
            for i, max_index in enumerate(row_max_indices):
                inner_index = max_index[0]
                if inner_index == qa_labels[i]:
                    step_hit_num += 1

        return (step_loss, step_shoot_num, step_hit_num)

    # cross模型的训练步
    def __match_train_step_for_cross(self, batch, optimizer, now_batch_num, scheduler, **kwargs):
        cross_entropy_function = nn.CrossEntropyLoss()

        input_ids = batch['input_ids'].to(self.device)
        token_type_ids = batch['token_type_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        batch_num, candidate_num, sequence_len = input_ids.shape

        input_ids = input_ids.reshape(batch_num * candidate_num, -1)
        token_type_ids = token_type_ids.reshape(batch_num * candidate_num, -1)
        attention_mask = attention_mask.reshape(batch_num * candidate_num, -1)

        # 得到模型的结果
        logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask).reshape(batch_num, candidate_num)

        qa_labels = torch.tensor([self.train_candidate_num-1]*logits.shape[0], dtype=torch.long, device=logits.device)
        # 计算损失
        step_loss = cross_entropy_function(logits, qa_labels)

        # 误差反向传播
        if not APEX_FLAG or self.no_apex:
            step_loss.backward()
        else:
            with amp.scale_loss(step_loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        # 更新模型参数
        if (now_batch_num + 1) % self.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # 统计R@1
        step_shoot_num = logits.shape[0]
        with torch.no_grad():
            _, row_max_indices = logits.topk(k=1, dim=-1)
            step_hit_num = (row_max_indices == (self.train_candidate_num-1)).sum().item()

        return (step_loss, step_shoot_num, step_hit_num)

    def __val_step_for_qa_input(self, batch):
        # 读取数据
        a_input_ids = (batch['a_input_ids']).to(self.device)
        a_token_type_ids = (batch['a_token_type_ids']).to(self.device)
        a_attention_mask = (batch['a_attention_mask']).to(self.device)

        b_input_ids = (batch['b_input_ids']).to(self.device)
        b_token_type_ids = (batch['b_token_type_ids']).to(self.device)
        b_attention_mask = (batch['b_attention_mask']).to(self.device)

        with torch.no_grad():
            # 得到模型的结果
            logits = self.model(
                a_input_ids=a_input_ids, a_token_type_ids=a_token_type_ids,
                a_attention_mask=a_attention_mask,
                b_input_ids=b_input_ids, b_token_type_ids=b_token_type_ids,
                b_attention_mask=b_attention_mask, no_aggregator=self.no_aggregator,
                no_enricher=self.no_enricher)

        return logits

    def __classify_val_step_for_cross(self, batch):
        # 读取数据
        input_ids = (batch['input_ids']).to(self.device)
        token_type_ids = (batch['token_type_ids']).to(self.device)
        attention_mask = (batch['attention_mask']).to(self.device)

        with torch.no_grad():
            # 得到模型的结果
            logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                                attention_mask=attention_mask)

        return logits

    def __match_val_step_for_bi(self, batch):
        # 读取数据
        # （batch_num, dim)
        a_input_ids = batch['a_input_ids'].to(self.device)
        a_token_type_ids = batch['a_token_type_ids'].to(self.device)
        a_attention_mask = batch['a_attention_mask'].to(self.device)

        b_input_ids = batch['b_input_ids'].to(self.device)
        b_token_type_ids = batch['b_token_type_ids'].to(self.device)
        b_attention_mask = batch['b_attention_mask'].to(self.device)

        # 得到模型的结果
        logits = self.model(
            a_input_ids=a_input_ids, a_token_type_ids=a_token_type_ids,
            a_attention_mask=a_attention_mask,
            b_input_ids=b_input_ids, b_token_type_ids=b_token_type_ids,
            b_attention_mask=b_attention_mask, train_flag=False, no_aggregator=self.no_aggregator,
            no_enricher=self.no_enricher)

        return logits

    def __match_val_step_for_cross(self, batch):
        # 读取数据
        # （batch_num, candidate_num, dim)
        input_ids = batch['input_ids'].to(self.device)
        token_type_ids = batch['token_type_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        batch_num, candidate_num, sequence_len = input_ids.shape

        input_ids = input_ids.reshape(batch_num*candidate_num, -1)
        token_type_ids = token_type_ids.reshape(batch_num*candidate_num, -1)
        attention_mask = attention_mask.reshape(batch_num*candidate_num, -1)

        # 得到模型的结果
        logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask).reshape(batch_num, candidate_num)

        return logits

    def __tokenize_classify_bi_data_then_save(self, data, save_name, a_column_name="premise", b_column_name="hypothesis", label_column_name='label'):
        # 读取数据到内存
        all_a_text = data[a_column_name]
        all_b_text = data[b_column_name]
        all_labels = data[label_column_name]
        all_index = data['idx']

        if self.model_class in ['MatchDeformer', 'ClassifyDeformer', 'MatchCrossBERT']:
            first_seq_max_len = self.first_seq_max_len
            second_seq_max_len = self.text_max_len - self.first_seq_max_len
        else:
            first_seq_max_len, second_seq_max_len = self.text_max_len, self.text_max_len

        # tokenize
        encoded_a_text = self.tokenizer(
            all_a_text, padding=True, verbose=False, add_special_tokens=True,
            truncation=True, max_length=first_seq_max_len, return_tensors='pt')

        encoded_b_text = self.tokenizer(
            all_b_text, padding=True, verbose=False, add_special_tokens=True,
            truncation=True, max_length=second_seq_max_len, return_tensors='pt')

        all_labels = all_labels
        dataset = DoubleInputLabelDataset(a_input_ids=encoded_a_text['input_ids'],
                                          a_attention_mask=encoded_a_text['attention_mask'],
                                          a_token_type_ids=encoded_a_text['token_type_ids'],
                                          b_input_ids=encoded_b_text['input_ids'],
                                          b_token_type_ids=encoded_b_text['token_type_ids'],
                                          b_attention_mask=encoded_b_text['attention_mask'],
                                          idx=torch.tensor(all_index), label=torch.tensor(all_labels))

        if not os.path.exists(root_path + "dataset"):
            os.makedirs(root_path + "dataset")

        # dataset.save_to_disk(root_path + "dataset/" + save_name)
        torch.save({'dataset': dataset}, root_path + "dataset/" + save_name)
        print(f"Processed dataset is saved at ./dataset/{save_name}")

        return dataset

    def __tokenize_classify_cross_data_then_save(self, data, save_name, a_column_name="premise", b_column_name="hypothesis", label_column_name='label'):
        # 读取数据到内存
        all_a_text = data[a_column_name]
        all_b_text = data[b_column_name]
        all_labels = data[label_column_name]
        all_index = data['idx']

        # tokenize
        all_texts = []
        for index, a_text in enumerate(all_a_text):
            all_texts.append((a_text, all_b_text[index]))

        # tokenize
        encoded_texts = self.tokenizer(
            all_texts, padding=True, verbose=False, add_special_tokens=True,
            truncation=True, max_length=self.text_max_len, return_tensors='pt')

        all_labels = all_labels

        dataset = SingleInputLabelDataset(input_ids=encoded_texts['input_ids'],
                                          token_type_ids=encoded_texts['token_type_ids'],
                                          attention_mask=encoded_texts['attention_mask'],
                                          idx=torch.tensor(all_index), label=torch.tensor(all_labels))

        if not os.path.exists(root_path + "dataset/"):
            os.makedirs(root_path + "dataset/")

        torch.save({'dataset': dataset}, root_path + "dataset/" + save_name)

        print(f"Processed dataset is saved at ./dataset/{save_name}")

        return dataset

    def __tokenize_match_bi_data_then_save(self, data, save_name, a_column_name="premise", b_column_name="hypothesis"):
        split_num = 100
        this_dataset_max_len = -1

        if self.model_class in ['MatchDeformer', 'ClassifyDeformer', 'MatchCrossBERT']:
            first_seq_max_len = self.first_seq_max_len
            second_seq_max_len = self.text_max_len - self.first_seq_max_len
        else:
            first_seq_max_len, second_seq_max_len = self.text_max_len, self.text_max_len

        # 读取数据到内存
        all_a_text = data[a_column_name]
        all_b_text = data[b_column_name]
        all_index = data['idx']

        # tokenize query block by block, context should be truncated from head ----------------------------
        query_step = math.ceil(len(all_a_text) / split_num)
        iter_num = math.ceil(len(all_a_text) / query_step)

        final_a_input_ids, final_a_token_type_ids, final_a_attention_mask = [], [], []
        for i in trange(iter_num):
            # tokenize this block
            this_block = all_a_text[i * query_step:(i + 1) * query_step]

            this_block_input_ids_list, this_block_token_type_ids_list, this_block_attention_mask_list, this_seq_max_len \
                = tokenize_and_truncate_from_head(self.tokenizer, this_block, first_seq_max_len)

            # update
            this_dataset_max_len = this_seq_max_len if this_dataset_max_len < this_seq_max_len else this_dataset_max_len
            final_a_input_ids += this_block_input_ids_list
            final_a_token_type_ids += this_block_token_type_ids_list
            final_a_attention_mask += this_block_attention_mask_list

        final_a_input_ids = torch.cat(final_a_input_ids, dim=0)
        final_a_token_type_ids = torch.cat(final_a_token_type_ids, dim=0)
        final_a_attention_mask = torch.cat(final_a_attention_mask, dim=0)

        # tokenize response in common practice ----------------------------
        encoded_b_text = self.tokenizer(
            all_b_text, padding=True, verbose=False, add_special_tokens=True,
            truncation=True, max_length=second_seq_max_len, return_tensors='pt')

        dataset = DoubleInputDataset(a_input_ids=final_a_input_ids, a_attention_mask=final_a_attention_mask, a_token_type_ids=final_a_token_type_ids,
                                     b_input_ids=encoded_b_text['input_ids'],  b_token_type_ids=encoded_b_text['token_type_ids'],
                                     b_attention_mask=encoded_b_text['attention_mask'], idx=torch.tensor(all_index))

        if not os.path.exists(root_path + "dataset"):
            os.makedirs(root_path + "dataset")

        torch.save({'dataset': dataset}, root_path + "dataset/" + save_name)

        print(f"Processed dataset is saved at ./dataset/{save_name}")
        print(f'context_max_len: {this_dataset_max_len}')
        print("*" * 20 + f"Encoding {final_a_input_ids.shape} texts finished!" + "*" * 20)

        return dataset

    def __tokenize_match_multi_candidate_data_then_save(self, data, save_name, a_column_name="sentence_a", b_column_name="candidates", candidate_num=100):
        if self.model_class in ['ClassifyDeformer']:
            first_seq_max_len = self.first_seq_max_len
            second_seq_max_len = self.text_max_len - self.first_seq_max_len
        else:
            first_seq_max_len, second_seq_max_len = self.text_max_len, self.text_max_len

        # avoid out of memory
        split_num = 10
        this_dataset_max_len = -1

        # 读取数据到内存
        all_a_text = data[a_column_name]
        all_candidates_lists = data[b_column_name]
        all_index = data['idx']

        from_end = False
        if self.dataset_name in ['msmarco', 'hard_msmarco']:
            pids = data['pids']
            from_end = True
            first_seq_max_len, second_seq_max_len = 32, 180
        else:
            pids = None
        print(f"first_seq_max_len: {first_seq_max_len}, second_seq_max_len: {second_seq_max_len}")


        # tokenize block by block, context should be truncated from head
        query_step = math.ceil(len(all_a_text) / split_num)
        iter_num = math.ceil(len(all_a_text) / query_step)

        final_a_input_ids, final_a_token_type_ids, final_a_attention_mask = [], [], []
        for i in trange(iter_num):
            # tokenize this block
            this_block = all_a_text[i * query_step:(i + 1) * query_step]

            this_block_input_ids_list, this_block_token_type_ids_list, this_block_attention_mask_list, this_seq_max_len \
                = tokenize_and_truncate_from_head(self.tokenizer, this_block, first_seq_max_len, from_end=from_end)

            # update
            this_dataset_max_len = this_seq_max_len if this_dataset_max_len < this_seq_max_len else this_dataset_max_len
            final_a_input_ids += this_block_input_ids_list
            final_a_token_type_ids += this_block_token_type_ids_list
            final_a_attention_mask += this_block_attention_mask_list

        final_a_input_ids = torch.cat(final_a_input_ids, dim=0)
        final_a_token_type_ids = torch.cat(final_a_token_type_ids, dim=0)
        final_a_attention_mask = torch.cat(final_a_attention_mask, dim=0)

        # tokenize candidates
        all_candidates = None
        actual_candidate_num = []
        actual_pids = []

        for index, this_candidates in enumerate(all_candidates_lists):
            # must include best answer
            if all_candidates is None:
                all_candidates = this_candidates[-candidate_num:]
            else:
                all_candidates += this_candidates[-candidate_num:]

            # get pids
            this_pids = []
            if self.dataset_name in ["msmarco", 'hard_msmarco']: 
                this_pids = pids[index][-candidate_num:] 

            # padding with "" and -1
            if len(this_candidates) < candidate_num:
                all_candidates += [""]*(candidate_num - len(this_candidates))

                if self.dataset_name in ["msmarco", "hard_msmarco"]:
                    this_pids += [-1]*(candidate_num - len(this_candidates))

            # save pid and candidate num
            if self.dataset_name in ["msmarco", "hard_msmarco"]:
                actual_pids.append(this_pids)

            actual_candidate_num.append(len(this_candidates))

        assert len(all_candidates) == candidate_num*len(all_a_text)

        # tokenize block by block
        split_num = 1000

        candidate_step = math.ceil(len(all_candidates) / split_num)
        iter_num = math.ceil(len(all_candidates) / candidate_step)

        final_candidate_input_ids, final_candidate_token_type_ids, final_candidate_attention_mask = [], [], []
        for i in trange(iter_num):
            # tokenize this block
            this_block = all_candidates[i * candidate_step:(i + 1) * candidate_step]

            this_block_input_ids_list, this_block_token_type_ids_list, this_block_attention_mask_list, this_seq_max_len \
                = tokenize_and_truncate_from_head(self.tokenizer, this_block, second_seq_max_len, from_end=True)

            # update
            this_dataset_max_len = this_seq_max_len if this_dataset_max_len < this_seq_max_len else this_dataset_max_len
            final_candidate_input_ids += this_block_input_ids_list
            final_candidate_token_type_ids += this_block_token_type_ids_list
            final_candidate_attention_mask += this_block_attention_mask_list

        final_candidate_input_ids = torch.cat(final_candidate_input_ids, dim=0)
        final_candidate_token_type_ids = torch.cat(final_candidate_token_type_ids, dim=0)
        final_candidate_attention_mask = torch.cat(final_candidate_attention_mask, dim=0)

        # save as dataset to disk
        if self.dataset_name in ['msmarco', "hard_msmarco"]:
            dataset = MSMARCODoubleInputDataset(a_input_ids=final_a_input_ids, a_attention_mask=final_a_attention_mask,
                                                a_token_type_ids=final_a_token_type_ids,
                                                b_input_ids=final_candidate_input_ids.view(-1, candidate_num,
                                                                                                 final_candidate_input_ids.shape[
                                                                                                     -1]),
                                                b_token_type_ids=final_candidate_token_type_ids.view(-1,
                                                                                                           candidate_num,
                                                                                                           final_candidate_token_type_ids.shape[
                                                                                                               -1]),
                                                b_attention_mask=final_candidate_attention_mask.view(-1,
                                                                                                           candidate_num,
                                                                                                           final_candidate_attention_mask.shape[
                                                                                                               -1]),
                                                idx=torch.tensor(all_index),
                                                actual_candidate_num=torch.tensor(actual_candidate_num),
                                                pids=torch.tensor(actual_pids))
        else:
            dataset = DoubleInputDataset(a_input_ids=final_a_input_ids, a_attention_mask=final_a_attention_mask,
                                         a_token_type_ids=final_a_token_type_ids,
                                         b_input_ids=final_candidate_input_ids.view(-1, candidate_num, final_candidate_input_ids.shape[-1]),
                                         b_token_type_ids=final_candidate_token_type_ids.view(-1, candidate_num, final_candidate_token_type_ids.shape[-1]),
                                         b_attention_mask=final_candidate_attention_mask.view(-1, candidate_num, final_candidate_attention_mask.shape[-1]),
                                         idx=torch.tensor(all_index))

        if not os.path.exists(root_path + "dataset"):
            os.makedirs(root_path + "dataset")

        torch.save({'dataset': dataset}, root_path + "dataset/" + save_name)

        print(f"Processed dataset is saved at ./dataset/{save_name}")
        print(f'context_max_len: {this_dataset_max_len}')
        print("*" * 20 + f"Encoding {final_a_input_ids.shape} texts finished!" + "*" * 20)

        return dataset

    def __tokenize_match_cross_data_then_save(self, data, save_name, a_column_name="sentence_a", b_column_name="candidates", idx_column_name="idx", candidate_num=100, suffix=""):
        """
        :param data: (idx, query, candidate_pool)
        :param save_name:
        :param a_column_name: query_column_name
        :param b_column_name: candidate_pool_column_name
        :param candidate_num: retrieve how many candidates from pool
        :param suffix:
        :return: SingleInputDataset: input ids---(query num, candidate num, seq len), attention mask....
        """
        # hyper-parameters
        this_dataset_max_len = -2
        split_num = 1000

        # 读取数据到内存
        all_a_text = data[a_column_name]
        all_candidates_lists = data[b_column_name]
        all_index = data[idx_column_name]

        if candidate_num > len(all_candidates_lists[0]):
            raise Exception(f'Candidate num is larger than {len(all_candidates_lists[0])}')

        # collect cross data
        all_texts = []
        for this_a, this_candidates in zip(all_a_text, all_candidates_lists):
            truncated_candidates = this_candidates[-candidate_num:]
            for candidate in truncated_candidates:
                all_texts.append((this_a, candidate))

        assert len(all_texts) == candidate_num*len(all_a_text)
        print("*"*20 + f"begin encoding {len(all_texts)} texts!" + "*"*20)

        # tokenize block by block
        query_step = math.ceil(len(all_a_text) / split_num)
        iter_num = math.ceil(len(all_a_text) / query_step)

        final_input_ids, final_token_type_ids, final_attention_mask = [], [], []
        for i in trange(iter_num):
            # tokenize this block
            this_block = all_texts[i*query_step*candidate_num:(i+1)*query_step*candidate_num]

            this_block_input_ids_list, this_block_token_type_ids_list, this_block_attention_mask_list, this_seq_max_len \
                = tokenize_and_truncate_from_head(self.tokenizer, this_block, self.text_max_len)

            # update
            this_dataset_max_len = this_seq_max_len if this_dataset_max_len < this_seq_max_len else this_dataset_max_len
            final_input_ids += this_block_input_ids_list
            final_token_type_ids += this_block_token_type_ids_list
            final_attention_mask += this_block_attention_mask_list

        final_input_ids = torch.cat(final_input_ids, dim=0)
        final_token_type_ids = torch.cat(final_token_type_ids, dim=0)
        final_attention_mask = torch.cat(final_attention_mask, dim=0)

        # print max response len
        max_response_len = torch.max(final_token_type_ids.sum(dim=-1))
        if max_response_len == self.text_max_len:
            print("Warning: Longest Response Occupy all input positions!")

        final_input_ids = final_input_ids.reshape(len(all_a_text), candidate_num, self.text_max_len)
        final_token_type_ids = final_token_type_ids.reshape(len(all_a_text), candidate_num, self.text_max_len)
        final_attention_mask = final_attention_mask.reshape(len(all_a_text), candidate_num, self.text_max_len)

        # set first token as cls
        final_input_ids[:, :, 0] = self.tokenizer.convert_tokens_to_ids('[CLS]')

        dataset = SingleInputDataset(input_ids=final_input_ids, token_type_ids=final_token_type_ids, attention_mask=final_attention_mask, idx=torch.tensor(all_index))

        if not os.path.exists(root_path + "dataset"):
            os.makedirs(root_path + "dataset")

        torch.save({'dataset': dataset}, root_path + "dataset/" + save_name + suffix)

        print(f"Processed dataset is saved at ./dataset/{save_name + suffix}")
        print(f'this_dataset_max_len: {this_dataset_max_len}')
        print("*" * 20 + f"Encoding {final_input_ids.shape} texts finished!" + "*" * 20)

        return dataset
