import math
from tqdm import trange
from transformers import AutoTokenizer
import torch
import datetime
import random


def print_message(*s, condition=True):
	s = ' '.join([str(x) for x in s])
	msg = "[{}] {}".format(datetime.datetime.now().strftime("%b %d, %H:%M:%S"), s)

	if condition:
		print(msg, flush=True)

	return msg


def tokenize_block_by_block(tokenizer, data, max_len):
	split_num = 1000
	query_step = math.ceil(len(data) / split_num)
	iter_num = math.ceil(len(data) / query_step)

	final_query_input_ids, final_query_token_type_ids, final_query_attention_mask = [], [], []
	for i in trange(iter_num):
		# tokenize this block
		this_block = data[i * query_step:(i + 1) * query_step]

		encoded_candidates = tokenizer(
			this_block, padding='max_length', verbose=False, add_special_tokens=True,
			truncation=True, max_length=max_len, return_tensors='pt')

		# update
		final_query_input_ids.append(encoded_candidates['input_ids'])
		final_query_token_type_ids.append(encoded_candidates['token_type_ids'])
		final_query_attention_mask.append(encoded_candidates['attention_mask'])

	final_query_input_ids = torch.cat(final_query_input_ids, dim=0)
	final_query_token_type_ids = torch.cat(final_query_token_type_ids, dim=0)
	final_query_attention_mask = torch.cat(final_query_attention_mask, dim=0)

	return {'input_ids': final_query_input_ids,
			'token_type_ids': final_query_token_type_ids,
			'attention_mask': final_query_attention_mask}


def tokenize_msmarco_train(tokenizer):
	print_message("Begin Process File!!!")

	# max len same as colbert
	query_max_len = 32
	doc_max_len = 180

	# begin to read data
	print_message("begin read data!!!")
	all_query, all_pos, all_neg = [], [], []

	train_file = open("../msmarco/triples.train.small.tsv", mode='r', encoding="utf-8")
	for index, line in enumerate(train_file):
		query, pos, neg = line.strip().split('\t')
		all_query.append(query)
		all_pos.append(pos)
		all_neg.append(neg)

		if index == 0:
			print("query:\n\t{}".format(query))
			print("positive:\n\t{}".format(pos))
			print("negative:\n\t{}".format(neg))

	train_file.close()

	assert len(all_query) == len(all_neg)
	assert len(all_neg) == len(all_pos)
	print_message(len(all_query))

	# sample a subset
	pool = range(len(all_query))
	sampled_index = random.sample(pool, 32 * 200000)

	sampled_query = []
	sampled_pos = []
	sampled_neg = []

	for index in sampled_index:
		sampled_query.append(all_query[index])
		sampled_neg.append(all_neg[index])
		sampled_pos.append(all_pos[index])

	print_message("Sample {} from {}!".format(len(sampled_query), len(all_query)))

	all_query = sampled_query
	all_neg = sampled_neg
	all_pos = sampled_pos

	# begin tokenize
	print_message("begin tokenize query!!!!!")
	tokenized_results = tokenize_block_by_block(tokenizer, all_query, query_max_len)
	torch.save(tokenized_results, "../../dataset/query_tokenized_dict")

	print_message("begin tokenize positive!!!!!")
	tokenized_results = tokenize_block_by_block(tokenizer, all_pos, doc_max_len)
	torch.save(tokenized_results, "../../dataset/positive_tokenized_dict")

	print_message("begin tokenize negative!!!!!")
	tokenized_results = tokenize_block_by_block(tokenizer, all_neg, doc_max_len)
	torch.save(tokenized_results, "../../dataset/negative_tokenized_dict")


def tokenize_msmarco_topk(topK_path, save_path):
	# begin to read data
	print_message("begin read data!!!")
	queries = {}
	topK_docs = {}
	topK_pids = {}

	print_message("#> Loading the top-k per query from", topK_path, "...")

	with open(topK_path) as f:
		for line_idx, line in enumerate(f):
			qid, pid, query, passage = line.split('\t')
			qid, pid = int(qid), int(pid)

			assert (qid not in queries) or (queries[qid] == query)
			queries[qid] = query
			topK_docs[qid] = topK_docs.get(qid, [])
			topK_docs[qid].append(passage)
			topK_pids[qid] = topK_pids.get(qid, [])
			topK_pids[qid].append(pid)

			print(f"\r{line_idx}", end="")
	print()

	idx, sentence_a, candidates, save_pids = [], [], [], []
	for index, key in enumerate(queries):
		idx.append(key)
		sentence_a.append(queries[key])
		candidates.append(topK_docs[key])
		save_pids.append(topK_pids[key])
		print(f"\r{index}", end="")
	print()

	print("get ready!")
	data_dict = {'idx':idx, 'sentence_a':sentence_a,
				 'candidates':candidates, 'pids':save_pids}
	torch.save(data_dict, save_path)


if __name__ == '__main__':
	this_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

	tokenize_msmarco_train(this_tokenizer)
	tokenize_msmarco_topk("../msmarco/top1000.dev", "../../dataset/string_msmarco_top1000_dev")
