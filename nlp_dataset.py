import random

import torch.utils.data


class SingleInputDataset(torch.torch.utils.data.Dataset):
	def __init__(self, input_ids, token_type_ids, attention_mask, idx):
		self.input_ids = input_ids
		self.token_type_ids = token_type_ids
		self.attention_mask = attention_mask
		self.idx = idx

	def __len__(self):  # 返回整个数据集的大小
		return len(self.input_ids)

	def __getitem__(self, index):
		item_dic = {'input_ids': self.input_ids[index],
					'token_type_ids': self.token_type_ids[index],
					'attention_mask': self.attention_mask[index],
					'idx': self.idx[index]}

		return item_dic


class DoubleInputDataset(torch.torch.utils.data.Dataset):
	def __init__(self, a_input_ids, a_token_type_ids, a_attention_mask,
				 b_input_ids, b_token_type_ids, b_attention_mask, idx):

		self.a_input_ids = a_input_ids
		self.a_token_type_ids = a_token_type_ids
		self.a_attention_mask = a_attention_mask
		self.b_input_ids = b_input_ids
		self.b_token_type_ids = b_token_type_ids
		self.b_attention_mask = b_attention_mask
		self.idx = idx

	def __len__(self):  # 返回整个数据集的大小
		return len(self.a_input_ids)

	def __getitem__(self, index):
		item_dic = {'a_input_ids': self.a_input_ids[index],
					'a_token_type_ids': self.a_token_type_ids[index],
					'a_attention_mask': self.a_attention_mask[index],
					'b_input_ids': self.b_input_ids[index],
					'b_token_type_ids': self.b_token_type_ids[index],
					'b_attention_mask': self.b_attention_mask[index],
					'idx': self.idx[index]}

		return item_dic


class MSMARCODoubleInputDataset(torch.torch.utils.data.Dataset):
	def __init__(self, a_input_ids, a_token_type_ids, a_attention_mask,
				 b_input_ids, b_token_type_ids, b_attention_mask, idx, actual_candidate_num, pids):
		"""
		in this dataset, all b_input_ids should have the same shape, however, in msmarco, each query may have different
		number of candidates, thus we should add empty candidates, thus need candidate_num
		"""
		self.a_input_ids = a_input_ids
		self.a_token_type_ids = a_token_type_ids
		self.a_attention_mask = a_attention_mask
		self.b_input_ids = b_input_ids
		self.b_token_type_ids = b_token_type_ids
		self.b_attention_mask = b_attention_mask
		self.actual_candidate_num = actual_candidate_num
		self.pids = pids
		self.idx = idx

	def __len__(self):  # 返回整个数据集的大小
		return len(self.a_input_ids)

	def __getitem__(self, index):
		item_dic = {'a_input_ids': self.a_input_ids[index],
					'a_token_type_ids': self.a_token_type_ids[index],
					'a_attention_mask': self.a_attention_mask[index],
					'b_input_ids': self.b_input_ids[index],
					'b_token_type_ids': self.b_token_type_ids[index],
					'b_attention_mask': self.b_attention_mask[index],
					'pid': self.pids[index],
					'candidate_num': self.actual_candidate_num[index],
					'idx': self.idx[index]}

		return item_dic

class DoubleInputLabelDataset(torch.torch.utils.data.Dataset):
	def __init__(self, a_input_ids, a_token_type_ids, a_attention_mask,
				 b_input_ids, b_token_type_ids, b_attention_mask, label, idx):

		self.a_input_ids = a_input_ids
		self.a_token_type_ids = a_token_type_ids
		self.a_attention_mask = a_attention_mask
		self.b_input_ids = b_input_ids
		self.b_token_type_ids = b_token_type_ids
		self.b_attention_mask = b_attention_mask
		self.label = label
		self.idx = idx

	def __len__(self):  # 返回整个数据集的大小
		return len(self.a_input_ids)

	def __getitem__(self, index):
		item_dic = {'a_input_ids': self.a_input_ids[index],
					'a_token_type_ids': self.a_token_type_ids[index],
					'a_attention_mask': self.a_attention_mask[index],
					'b_input_ids': self.b_input_ids[index],
					'b_token_type_ids': self.b_token_type_ids[index],
					'b_attention_mask': self.b_attention_mask[index],
					'label': self.label[index],
					'idx': self.idx[index]}

		return item_dic


class MSMARCODataset(torch.torch.utils.data.Dataset):
	def __init__(self, queries_ids, qid_2_tensors, qid_2_negs, qid_2_pos, pid_2_tensors):
		self.queries_ids = queries_ids

		self.qid_2_tensors = qid_2_tensors
		self.qid_2_negs = qid_2_negs
		self.qid_2_pos = qid_2_pos

		self.pid_2_tensors = pid_2_tensors

		for qid in self.queries_ids:
			random.shuffle(self.qid_2_negs[qid])

	def get_tensor_by_id(self, tensor_dict, input_id):
		return tensor_dict[input_id]['input_ids'], tensor_dict[input_id]['attention_mask'], \
			   tensor_dict[input_id]['token_type_ids']

	def __getitem__(self, item):
		query_id = self.queries_ids[item]
		a_input_ids, a_attention_mask, a_token_type_ids = self.get_tensor_by_id(self.qid_2_tensors, query_id)

		pos_id = self.qid_2_pos[query_id].pop(0)  # Pop positive and add at end
		b_input_ids, b_attention_mask, b_token_type_ids = self.get_tensor_by_id(self.pid_2_tensors, pos_id)
		self.qid_2_pos[query_id].append(pos_id)

		neg_id = self.qid_2_negs[query_id].pop(0)  # Pop negative and add at end
		c_input_ids, c_attention_mask, c_token_type_ids = self.get_tensor_by_id(self.pid_2_tensors, neg_id)
		self.qid_2_negs[query_id].append(neg_id)

		item_dic = {'a_input_ids': a_input_ids,
					'a_attention_mask': a_attention_mask,
					'a_token_type_ids': a_token_type_ids,
					'b_input_ids': b_input_ids,
					'b_attention_mask': b_attention_mask,
					'b_token_type_ids': b_token_type_ids,
					'c_input_ids': c_input_ids,
					'c_attention_mask': c_attention_mask,
					'c_token_type_ids': c_token_type_ids,
					'idx': query_id}

		return item_dic

	def __len__(self):
		return len(self.queries_ids)


class SingleInputLabelDataset(torch.torch.utils.data.Dataset):
	def __init__(self, input_ids, token_type_ids, attention_mask, label, idx):
		self.input_ids = input_ids
		self.token_type_ids = token_type_ids
		self.attention_mask = attention_mask
		self.label = label
		self.idx = idx

	def __len__(self):  # 返回整个数据集的大小
		return len(self.input_ids)

	def __getitem__(self, index):
		item_dic = {'input_ids': self.input_ids[index],
					'token_type_ids': self.token_type_ids[index],
					'attention_mask': self.attention_mask[index],
					'label': self.label[index],
					'idx': self.idx[index]}

		return item_dic