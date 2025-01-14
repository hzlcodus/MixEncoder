import argparse
import os
import time

from nlp_trainer import TrainWholeModel
from my_function import get_elapse_time, set_seed

root_path = "/data/cme/fast_match/"

def read_arguments():
	parser = argparse.ArgumentParser()

	# default arguments
	parser.add_argument("--seed", "-s", default=42, type=int)
	parser.add_argument("--text_max_len", default=512, type=int)
	parser.add_argument("--num_train_epochs", "-e", type=int, default=50)

	# must set
	# add model
	parser.add_argument("--model_class", required=True, type=str)

	# related to data
	parser.add_argument("--dataset_name", "-d", type=str, choices=['dstc7', 'mnli', 'ubuntu', 'qqp', 'yahooqa',
																   'boolq', 'msmarco', 'hard_msmarco'])
	parser.add_argument("--label_num", required=True, type=int, help="for match task, please set as 1")

	# related to model
	parser.add_argument("--composition", type=str, default='pooler',
						help='control the way to get sentence representation')
	parser.add_argument("--context_num", "-c", default=1, type=int)
	parser.add_argument("--pretrained_bert_path", default='prajjwal1/bert-small', type=str)
	parser.add_argument("--model_save_prefix", default="", type=str)

	parser.add_argument("--top_layer_num", default=3, type=int, help='used for deformer')
	parser.add_argument("--first_seq_max_len", default=256, type=int, help="only need by deformer")

	parser.add_argument("--train_candidate_num", default=-1, type=int, help="only need by cross, according to hardware")
	parser.add_argument("--used_layers", type=str, help="If None, use the last layer, if 0, do not use layer,\
     if a,b,c..., use a-th, b-th layers, if a-b, use layer [a,b].")
 
	# related to train
	parser.add_argument("--one_stage", action="store_true", default=False)
	parser.add_argument("--evaluate_epoch", default=1, type=int)
	parser.add_argument("--no_aggregator", action="store_true", default=False, help="use to enable Parallel-Encoder ablation")
	parser.add_argument("--no_enricher", action="store_true", default=False, help="use to enable Parallel-Encoder ablation")
	parser.add_argument("--no_apex", action="store_true", default=False)
	parser.add_argument("--no_train", action="store_true", default=False)
	parser.add_argument("--do_test", action="store_true", default=False)
	parser.add_argument("--do_real_test", action="store_true", default=False)
	parser.add_argument("--query_block_size", default=200, type=int, help="only need by match task")
	parser.add_argument("--do_val", action="store_true", default=False)
	parser.add_argument("--use_cpu", action="store_true", default=False)
	parser.add_argument("--in_batch", action="store_true", default=False)
	parser.add_argument("--nvidia_number", "-n", required=True, type=str)
	parser.add_argument("--restore", action="store_true", default=False,
						help="use restore and only_final together to control which model to read!")

	parser.add_argument("--val_batch_size", default=64, type=int, help="control the batch size of val as well as test")
	parser.add_argument("--train_batch_size", default=64, type=int)
	parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
	parser.add_argument("--step_max_query_num", type=int, default=-1,
						help="necessary for __efficient_match_train_step_for_qa_input")
	parser.add_argument("--clean_graph", action="store_true", default=False,
						help="do forward which reduce cuda memory of graph but increase reference time.")

	parser.add_argument("--no_initial_test", action="store_true", default=False)
	parser.add_argument("--only_final", action="store_true", default=False,
						help="using two stage setting but only train last stage")

	parser.add_argument("--load_model", "-l", action="store_true", default=False)
	parser.add_argument("--load_model_path", type=str, help="load classifier")

	parser.add_argument("--save_model_dict", default="/data/cme/mix/model/", type=str)
	parser.add_argument("--last_model_dict", default="/data/cme/mix/last_model/", type=str)

	parser.add_argument("--load_middle", action="store_true", default=False)

	parser.add_argument("--distill", action="store_true", default=False)
	parser.add_argument("--teacher_path", default="./model/teacher", type=str)
	parser.add_argument("--mlm", action="store_true", default=False)

	# jcy : regularization, teacher loss
	parser.add_argument("--teacher_loss", action="store_true", default=False)
	parser.add_argument("--alpha", default=0.5, type=float)

	# 设置并行需要改的
	parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')
	parser.add_argument("--data_parallel", action="store_true", default=False)
	parser.add_argument("--data_distribute", action="store_true", default=False)

	# outdated
	parser.add_argument("--dataset_split_num", default=20, type=int)
	parser.add_argument("--first_stage_lr", default=0.3, type=float, help="the lr of memory at first stage")

	args = parser.parse_args()
	print("args:", args)
	return args


if __name__ == '__main__':
	my_args = read_arguments()
	my_args.no_apex = True

	# 设置随机种子 # 주석: seed 설정
	set_seed(my_args.seed)

	# begin time # 주석: 시작 시간
	begin_time = time.time()

	# 创建训练类 # 주석: TrainWholeModel 클래스 생성
	my_train_model = TrainWholeModel(my_args)

	# 设置训练参数 # 주석: 2 stage 학습을 위한 flag
	my_train_two_stage_flag = False 
	# add model
	if my_args.model_class in ['MatchParallelEncoder', 'CLSMatchParallelEncoder',
							   'CLSClassifyParallelEncoder', 'DisenCLSMatchParallelEncoder']:
		my_train_two_stage_flag = True

	if my_args.one_stage:
		my_train_two_stage_flag = False

	if my_args.distill:
		raise Exception("Distillation is not supported yes!")

	if not os.path.exists(my_args.save_model_dict):
		os.makedirs(my_args.save_model_dict)
	if not os.path.exists(my_args.last_model_dict):
		os.makedirs(my_args.last_model_dict)
	if not os.path.exists(root_path + "dataset/"):
		os.makedirs(root_path + "dataset/")

	# 训练
	if not my_args.no_train:
		my_train_model.train(train_two_stage_flag=my_train_two_stage_flag,
							 only_final=my_args.only_final)

	# 普通的评测，即将被淘汰
	if my_args.do_test or my_args.do_val:
		my_train_model.do_test(model_save_path=my_args.save_model_dict + "/" + my_args.model_save_prefix +
											   my_args.model_class + "_" +
											   my_args.dataset_name, do_val=my_args.do_val)

	# 测速 
	if my_args.do_real_test:
		if my_args.dataset_name in ['dstc7', 'ubuntu', 'msmarco']:
			if my_args.model_class in ['QAMatchModel', 'CMCModel', 'MatchParallelEncoder', 'PolyEncoder', 'MatchDeformer',
									   'DisenCLSMatchParallelEncoder', 'CLSMatchParallelEncoder', 'ColBERT']:
				my_train_model.match_bi_real_test(
					model_save_path=my_args.save_model_dict + "/" + my_args.model_save_prefix +
									my_args.model_class + "_" +
									my_args.dataset_name)
			elif my_args.model_class in ['MatchCrossBERT']:
				my_train_model.match_bi_real_test_for_match_cross(
					model_save_path=my_args.save_model_dict + "/" + my_args.model_save_prefix +
									my_args.model_class + "_" +
									my_args.dataset_name)
			else:
				raise Exception(f"{my_args.model_class} is not supported for real test yet!")
		elif my_args.dataset_name in ['mnli', 'qqp']:
			if my_args.model_class in ['QAClassifierModel', 'ClassifyParallelEncoder', 'PolyEncoder', 'ClassifyDeformer']:
				my_train_model.classify_bi_real_test(
					model_save_path=my_args.save_model_dict + "/" + my_args.model_save_prefix +
									my_args.model_class + "_" +
									my_args.dataset_name)
			elif my_args.model_class in ['CrossBERT']:
				my_train_model.classify_cross_real_test(
					model_save_path=my_args.save_model_dict + "/" + my_args.model_save_prefix +
									my_args.model_class + "_" +
									my_args.dataset_name)
			else:
				raise Exception(f"{my_args.model_class} is not supported for real test yet!")
		else:
			print(f"Warning: {my_args.dataset_name} is not supported for real test yet!")

	print("*" * 100)
	print("Finish training and take", get_elapse_time(begin_time))
	print("*" * 100)
