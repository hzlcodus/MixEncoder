import gzip
import math

from torch.utils.data import Dataset
from tqdm import trange
from transformers import AutoTokenizer
import torch
import datetime
import random
from typing import Dict, Tuple
from tqdm.autonotebook import tqdm
import json
import os
import logging
import csv

logger = logging.getLogger(__name__)


def tokenize_block_by_block(tokenizer, max_len, id_text_dict):
	ids = []
	text_data = []
	for this_id in id_text_dict:
		ids.append(this_id)
		text_data.append(id_text_dict[this_id])

	assert len(ids) == len(text_data)

	split_num = 1000
	query_step = math.ceil(len(text_data) / split_num)
	iter_num = math.ceil(len(text_data) / query_step)

	final_query_input_ids, final_query_token_type_ids, final_query_attention_mask = [], [], []
	for i in trange(iter_num):
		# tokenize this block
		this_block = text_data[i * query_step:(i + 1) * query_step]

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

	print(final_query_input_ids.shape)

	id_2_tensors = {}
	for index, this_id in enumerate(ids):
		id_2_tensors[this_id] = {'input_ids': final_query_input_ids[index],
								 'token_type_ids': final_query_token_type_ids[index],
								 'attention_mask': final_query_attention_mask[index]}

	return id_2_tensors


# copy from repo: beir-cellar/beir/
class GenericDataLoader:
	def __init__(self, data_folder: str = None, prefix: str = None, corpus_file: str = "corpus.jsonl",
				 query_file: str = "queries.jsonl",
				 qrels_folder: str = "qrels", qrels_file: str = ""):
		self.corpus = {}
		self.queries = {}
		self.qrels = {}

		if prefix:
			query_file = prefix + "-" + query_file
			qrels_folder = prefix + "-" + qrels_folder

		self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
		self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
		self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else None
		self.qrels_file = qrels_file

	@staticmethod
	def check(fIn: str, ext: str):
		if not os.path.exists(fIn):
			raise ValueError("File {} not present! Please provide accurate file.".format(fIn))

		if not fIn.endswith(ext):
			raise ValueError("File {} must be present with extension {}".format(fIn, ext))

	def load_custom(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

		self.check(fIn=self.corpus_file, ext="jsonl")
		self.check(fIn=self.query_file, ext="jsonl")
		self.check(fIn=self.qrels_file, ext="tsv")

		if not len(self.corpus):
			logger.info("Loading Corpus...")
			self._load_corpus()
			logger.info("Loaded %d Documents.", len(self.corpus))
			logger.info("Doc Example: %s", list(self.corpus.values())[0])

		if not len(self.queries):
			logger.info("Loading Queries...")
			self._load_queries()

		if os.path.exists(self.qrels_file):
			self._load_qrels()
			self.queries = {qid: self.queries[qid] for qid in self.qrels}
			logger.info("Loaded %d Queries.", len(self.queries))
			logger.info("Query Example: %s", list(self.queries.values())[0])

		return self.corpus, self.queries, self.qrels

	def load(self, split="test") -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

		self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
		self.check(fIn=self.corpus_file, ext="jsonl")
		self.check(fIn=self.query_file, ext="jsonl")
		self.check(fIn=self.qrels_file, ext="tsv")

		if not len(self.corpus):
			logger.info("Loading Corpus...")
			self._load_corpus()
			logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
			logger.info("Doc Example: %s", list(self.corpus.values())[0])

		if not len(self.queries):
			logger.info("Loading Queries...")
			self._load_queries()

		if os.path.exists(self.qrels_file):
			self._load_qrels()
			self.queries = {qid: self.queries[qid] for qid in self.qrels}
			logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
			logger.info("Query Example: %s", list(self.queries.values())[0])

		return self.corpus, self.queries, self.qrels

	def load_corpus(self) -> Dict[str, Dict[str, str]]:

		self.check(fIn=self.corpus_file, ext="jsonl")

		if not len(self.corpus):
			logger.info("Loading Corpus...")
			self._load_corpus()
			logger.info("Loaded %d Documents.", len(self.corpus))
			logger.info("Doc Example: %s", list(self.corpus.values())[0])

		return self.corpus

	def _load_corpus(self):

		num_lines = sum(1 for i in open(self.corpus_file, 'rb'))
		with open(self.corpus_file, encoding='utf8') as fIn:
			for line in tqdm(fIn, total=num_lines):
				line = json.loads(line)
				self.corpus[line.get("_id")] = {
					"text": line.get("text"),
					"title": line.get("title"),
				}

	def _load_queries(self):

		with open(self.query_file, encoding='utf8') as fIn:
			for line in fIn:
				line = json.loads(line)
				self.queries[line.get("_id")] = line.get("text")

	def _load_qrels(self):

		reader = csv.reader(open(self.qrels_file, encoding="utf-8"),
							delimiter="\t", quoting=csv.QUOTE_MINIMAL)
		next(reader)

		for id, row in enumerate(reader):
			query_id, corpus_id, score = row[0], row[1], int(row[2])

			if query_id not in self.qrels:
				self.qrels[query_id] = {corpus_id: score}
			else:
				self.qrels[query_id][corpus_id] = score


if __name__ == '__main__':
	this_tokenizer = AutoTokenizer.from_pretrained("../../bert-base-uncased")

	print("load corpus...", end=" ")
	corpus, queries, _ = GenericDataLoader("../hard_msmarco/msmarco/").load(split="train")
	print("finished!")

	msmarco_triplets_filepath = "../hard_msmarco/msmarco-hard-negatives.jsonl.gz"

	#################################
	#### Parameters for Training ####
	#################################

	ce_score_margin = 3
	num_negs_per_system = 5

	#### Load the hard negative MSMARCO jsonl triplets from SBERT
	#### These contain a ce-score which denotes the cross-encoder score for the query and passage.
	#### We chose a margin between positive and negative passage scores => above which consider negative as hard negative.
	#### Finally to limit the number of negatives per passage, we define num_negs_per_system across all different systems.

	train_queries = {}
	all_pos = 0
	all_neg = 0
	with gzip.open(msmarco_triplets_filepath, 'rt', encoding='utf8') as fIn:
		for line in tqdm(fIn, total=502939):
			data = json.loads(line)

			# Get the positive passage ids
			pos_pids = [item['pid'] for item in data['pos']]
			pos_min_ce_score = min([item['ce-score'] for item in data['pos']])
			ce_score_threshold = pos_min_ce_score - ce_score_margin

			# Get the hard negatives
			neg_pids = set()
			for system_negs in data['neg'].values():
				negs_added = 0
				for item in system_negs:
					if item['ce-score'] > ce_score_threshold:
						continue

					pid = item['pid']
					if pid not in neg_pids:
						neg_pids.add(pid)
						negs_added += 1
						if negs_added >= num_negs_per_system:
							break

			if len(pos_pids) > 0 and len(neg_pids) > 0:
				all_pos += len(pos_pids)
				all_neg += len(neg_pids)
				train_queries[data['qid']] = {'query': queries[data['qid']], 'pos': pos_pids,
											  'hard_neg': list(neg_pids)}

	logging.info("Train queries: {}".format(len(train_queries)))

	query_num = len(train_queries)
	print(f"all pos {all_pos}, all neg {all_neg}, avg pos {all_pos/query_num}, avg neg {all_neg/query_num}")

	#################################
	#### Encoding and Save ####
	#################################
	queries_ids = list(train_queries.keys())

	qid_2_negs = {}
	qid_2_pos = {}
	qid_2_text = {}
	all_pids = []
	for qid in queries_ids:
		qid_2_negs[qid] = train_queries[qid]['hard_neg']
		qid_2_pos[qid] = train_queries[qid]['pos']
		qid_2_text[qid] = train_queries[qid]['query']

		all_pids.extend(train_queries[qid]['pos'])
		all_pids.extend(train_queries[qid]['hard_neg'])


	qid_2_tensors = tokenize_block_by_block(this_tokenizer, 32, qid_2_text)

	pid_2_text = {}
	for pid in all_pids:
		pid_2_text[pid] = corpus[pid]["text"]

	pid_2_tensors = tokenize_block_by_block(this_tokenizer, 128, pid_2_text)

	# train_dataset = MSMARCODataset(queries_ids, qid_2_tensors, qid_2_negs, qid_2_pos, pid_2_tensors)

	save_dict = {'queries_ids': queries_ids, 'qid_2_tensors': qid_2_tensors,
				 'qid_2_negs': qid_2_negs, 'qid_2_pos': qid_2_pos,
				 'pid_2_tensors': pid_2_tensors}

	torch.save(save_dict, "../../dataset/tokenized_hard_msmarco_train")
