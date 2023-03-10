import argparse
import time
import os
import numpy as np
import torch
import json

import sys

sys.path.insert(0, './bao_server')
import bao_server.model as model
import csv
import copy
import random
from random import shuffle

sys.path.append('../ShiftHandler')
from replay_buffer import summarizer


def unnormalize_torch(vals, min_val, max_val):
	vals = (vals * (max_val - min_val)) + min_val
	return torch.exp(vals)


def get_query_list(x, y, num_train_per_task, num_test_per_task, is_imb=True, num_burnin=4, num_task=3, seed=0):
	file_name = "./train"
	num_queries_per_file = 100000

	x_filtered = []
	y_filtered = []

	np.random.seed(seed)
	random.seed(seed)

	joins = []
	predicates = []
	tables = []
	label = []
	numerical_label = []

	templates = {}
	template_joins = {}
	template_ids = []
	temp_counts = []

	with open(file_name + ".csv", 'rU') as f:
		data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
		for row in data_raw[:num_queries_per_file]:
			tables.append(row[0].split(','))
			joins.append(row[1].split(','))
			predicates.append(row[2].split(','))

			joined_tables = row[0].split(',')
			joined_tables.sort()
			if ','.join(joined_tables) not in templates:
				templates[','.join(joined_tables)] = len(templates)
				template_joins[templates[','.join(joined_tables)]] = len(joined_tables)
				temp_counts.append(0)
			template_ids.append(templates[','.join(joined_tables)])
			temp_counts[templates[','.join(joined_tables)]] += 1

			if int(row[3]) < 1:
				print("Queries must have non-zero cardinalities")
				exit(1)
			label.append(row[3])
			numerical_label.append(int(row[3]))
	print("Loaded queries")

	if is_imb:
		ratio_per_temp = [5, 20, 75]
	else:
		ratio_per_temp = [33, 33, 34]

	ratio_per_temp.reverse()

	filtered_temp_list_all = [i for i in range(len(temp_counts)) if temp_counts[i] >= 1100]
	filtered_temp_list = []

	mean_list = []
	for tmp_id in filtered_temp_list_all:
		total_id_per_tmp = [i for i in range(num_queries_per_file) if template_ids[i] == tmp_id]
		y_list = []
		for idx in total_id_per_tmp:
			y_list.append(numerical_label[idx])
		mean_list.append(np.mean(y_list))

	median_list = []
	for tmp_id in filtered_temp_list_all:
		total_id_per_tmp = [i for i in range(num_queries_per_file) if template_ids[i] == tmp_id]
		y_list = []
		for idx in total_id_per_tmp:
			y_list.append(numerical_label[idx])
		median_list.append(np.median(y_list))

	sort_index = [i for i, x in sorted(enumerate(median_list), key=lambda x: x[1])]
	num_per_category = int(len(sort_index)/num_task)
	task_candi_lens = [6, 8, 5]

	for i in range(num_task):
		task_candi = sort_index[sum(task_candi_lens[:i]):sum(task_candi_lens[:i+1])]
		chosen_idx = random.choice(task_candi)
		filtered_temp_list.append(filtered_temp_list_all[chosen_idx])
	median_list.sort()
	train_list = []
	test_list = []

	candi_train_id_list = []
	candi_test_id_list = []


	for ind_i, ratio_i in enumerate(ratio_per_temp):
		temp_id = filtered_temp_list[ind_i]
		total_id_per_tmp = [i for i in range(num_queries_per_file) if template_ids[i] == temp_id]
		shuffle(total_id_per_tmp)
		# total_id_per_tmp = total_id_per_tmp[0:-num_test_per_task]
		candi_train_id_list.append(total_id_per_tmp[0:-num_test_per_task])
		candi_test_id_list.append(total_id_per_tmp[-num_test_per_task:])

	x_train_tmp = []
	y_train_tmp = []
	for _ in range(num_burnin):
		for ind_i, ratio_i in enumerate(ratio_per_temp):
			num_queries_train_per_temp = int(ratio_per_temp[ind_i] * num_train_per_task / sum(ratio_per_temp))
			total_id_per_tmp = copy.deepcopy(candi_train_id_list[ind_i])
			shuffle(total_id_per_tmp)
			train_id_per_tmp = total_id_per_tmp[:num_queries_train_per_temp]

			x_train_tmp.extend([x[q_i] for q_i in train_id_per_tmp])
			y_train_tmp.extend([y[q_i] for q_i in train_id_per_tmp])

			if num_queries_train_per_temp > len(total_id_per_tmp):
				k_needed = num_queries_train_per_temp - len(total_id_per_tmp)
				sampled_per_tmp = random.choices(total_id_per_tmp, k=k_needed)
				x_train_addition = [x[q_i] for q_i in sampled_per_tmp]
				y_train_addition = [y[q_i] for q_i in sampled_per_tmp]

				x_train_tmp.extend(x_train_addition)
				y_train_tmp.extend(y_train_addition)

	x_filtered.extend(x_train_tmp)
	y_filtered.extend(y_train_tmp)

	for i in range(num_burnin):
		train_list.append((x_train_tmp[i * num_train_per_task: i * num_train_per_task + num_train_per_task],
		                   y_train_tmp[i * num_train_per_task: i * num_train_per_task + num_train_per_task]))

	for task_id in range(len(ratio_per_temp)):

		train_id_per_tmp = copy.deepcopy(candi_train_id_list[task_id])
		shuffle(train_id_per_tmp)
		train_id_per_tmp = train_id_per_tmp[:num_train_per_task]

		test_id_per_tmp = copy.deepcopy(candi_test_id_list[task_id])

		x_train = [x[q_i] for q_i in train_id_per_tmp]
		y_train = [y[q_i] for q_i in train_id_per_tmp]
		train_list.append((x_train, y_train))

		x_filtered.extend(x_train)
		y_filtered.extend(y_train)

		x_test = [x[q_i] for q_i in test_id_per_tmp]
		y_test = [y[q_i] for q_i in test_id_per_tmp]
		test_list.append((x_test, y_test))

		x_filtered.extend(x_test)
		y_filtered.extend(y_test)
	return x_filtered, y_filtered, train_list, test_list


def qerror_loss(preds, targets, min_val, max_val):
	qerror = []
	preds = unnormalize_torch(preds, min_val, max_val)
	targets = unnormalize_torch(targets, min_val, max_val)

	for i in range(len(targets)):
		if (preds[i] > targets[i]).cpu().data.numpy()[0]:
			qerror.append(preds[i] / targets[i])
		else:
			qerror.append(targets[i] / preds[i])
	return torch.mean(torch.cat(qerror))


def get_qerrors(preds_unnorm, labels_unnorm):
	qerror = []
	for i in range(len(preds_unnorm)):
		if preds_unnorm[i] > float(labels_unnorm[i]):
			qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
		else:
			qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))
	return qerror


def print_qerror(preds_unnorm, labels_unnorm):
	qerror = []
	for i in range(len(preds_unnorm)):
		if preds_unnorm[i] > float(labels_unnorm[i]):
			qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
		else:
			qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

	print("Median: {}".format(np.median(qerror)))
	print("90th percentile: {}".format(np.percentile(qerror, 90)))
	print("95th percentile: {}".format(np.percentile(qerror, 95)))
	print("99th percentile: {}".format(np.percentile(qerror, 99)))
	print("Max: {}".format(np.max(qerror)))
	print("Mean: {}".format(np.mean(qerror)))

def qerrors(preds, targets):
	qerror_res = []
	for i in range(len(targets)):
		if preds[i] > targets[i]:
			qerror_res.append(preds[i] / targets[i])
		else:
			qerror_res.append(targets[i] / preds[i])
	return qerror_res


def train_and_predict(x, y, train_list, test_list, cuda, buffer, buffer_size, num_tasks=6, concentration=1e-4,
                      tradeoff=0.5, seed=0):
	# Load training and validation data
	print("buffer: {}".format(buffer))
	print("seed: {}".format(seed))
	print("concentration: {}".format(concentration))
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)

	reg = model.BaoRegression(have_cache_data=True, verbose=False)
	reg.fit_feature_extractor(x, y)

	prev_performs = []
	best_performs = []

	latest_buffer = []
	if buffer == 'lwp':
		handler_buffer = summarizer(buffer_limit=buffer_size, loss_ada=True,
		                         concentration=concentration,
		                         is_move=False)
	else:
		handler_buffer = summarizer(buffer_limit=buffer_size, loss_ada=False,
		                         concentration=concentration,
		                         is_move=False)
	num_queries_seen_far = 0

	for task_id in range(num_tasks):
		# first replay old queries
		replay_list = []
		if buffer == 'cbp' or buffer.lower() == 'lwp':
			replay_queries_tmp, _ = handler_buffer.get_all_samples()
			for (_, y_i, plan, _, _) in replay_queries_tmp:
				replay_list.append((plan, y_i))
		elif buffer == 'latest':
			replay_list = latest_buffer
		elif buffer == 'rs':
			replay_list = latest_buffer
		elif buffer == 'all':
			replay_list = latest_buffer

		(x_train, y_train) = train_list[task_id]

		current_bs = len(x_train)
		x_train_all = copy.deepcopy(x_train)
		y_train_all = copy.deepcopy(y_train)

		for (plan, y_i) in replay_list:
			x_train_all.append(plan)
			y_train_all.append(y_i)

		ada_size = False
		if buffer.lower() == 'lwp':
			ada_size = True

		if tradeoff != 0 and buffer != 'latest' and len(replay_list) > 0:
			idx_list, current_losses, replay_idx_list, replay_losses_list = reg.fit_model(x_train_all, y_train_all,
			                                                                              tradeoff=tradeoff, ada_size=ada_size,
			                                                                              size_current_batch=current_bs, seed=seed)
		else:
			idx_list, current_losses, replay_idx_list, replay_losses_list = reg.fit_model(x_train_all, y_train_all, seed=seed,
			                                                                              ada_size=ada_size)

			# update buffer size based on loss
		if buffer == 'lwp' and len(replay_losses_list):
			norm_losses_list = [None] * handler_buffer.buffer_size
			for (q_id, loss) in zip(replay_idx_list, replay_losses_list):
				norm_losses_list[q_id] = loss[0]
			handler_buffer.update_losses(norm_losses_list)

		if task_id > 1:
			(x_test, y_test) = test_list[task_id - 2]
			preds = reg.predict(x_test)
			preds = np.squeeze(preds)
			square_error = qerrors(preds, y_test)

			res = {"buffer": buffer, "size": buffer_size, "concentration": concentration, "seed": seed,
			       "median": np.percentile(square_error, 50), "95": np.percentile(square_error, 95),
			       "max": np.max(square_error), "mean": np.mean(square_error)}
			best_performs.append(res)

			prev_perform_this_task = []
			for prev_task in range(3):
				(x_test, y_test) = test_list[prev_task]
				preds = reg.predict(x_test)
				preds = np.squeeze(preds)
				square_error = qerrors(preds, y_test)
				res = {"buffer": buffer, "size": buffer_size, "concentration": concentration, "seed": seed,
				       "median": np.percentile(square_error, 50), "95": np.percentile(square_error, 95),
				       "max": np.max(square_error), "mean": np.mean(square_error)}
				prev_perform_this_task.append(res)

			prev_performs.append(prev_perform_this_task)

		if task_id == num_tasks - 1:
			break

		# add new queries to the replay buffer
		(x_train, y_train) = train_list[task_id]
		if buffer == 'cbp' or buffer.lower() == 'lwp':
			experience_features = reg.get_before_features(x_train)
			for i in range(experience_features.shape[0]):
				q_feature = experience_features[i, :]
				if buffer == 'lwp':
					loss_id = idx_list.index(i)
					handler_buffer.process_a_query(q_feature,  y_train[i], x_train[i], None, current_losses[loss_id][0])
				else:
					handler_buffer.process_a_query(q_feature, y_train[i], x_train[i])
		elif buffer == 'latest':
			for i in range(len(x_train)):
				if len(latest_buffer) < buffer_size:
					latest_buffer.append((x_train[i], y_train[i]))
				else:
					latest_buffer.pop(0)
					latest_buffer.append((x_train[i], y_train[i]))
		elif buffer == 'rs':
			for i in range(len(x_train)):
				if len(latest_buffer) < buffer_size:
					latest_buffer.append((x_train[i], y_train[i]))
				else:
					random_i = random.uniform(0, 1)
					if random_i < float(len(latest_buffer)) / (num_queries_seen_far + 1):
						latest_buffer.pop(0)
						latest_buffer.append((x_train[i], y_train[i]))
				num_queries_seen_far += 1
		elif buffer == 'all':
			for i in range(len(x_train)):
				latest_buffer.append((x_train[i], y_train[i]))


	file_name = "./cost_result"
	with open(file_name, "a") as f:
		f.write("best\n")
		f.write("{}\n".format(json.dumps(best_performs)))
		f.write("prev\n")
		f.write("{}\n".format(json.dumps(prev_performs)))
		f.flush()
	print("best_performs")
	print(best_performs)
	print("prev_performs")
	print(prev_performs)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--imbalance', default=False, help="is imbalance?", action='store_true')
	parser.add_argument("--buffer", help="cbp, lwp, latest, rs", default="cbp")
	parser.add_argument("--buffersize", help="buffer size", type=int, default = 50)

	parser.add_argument("--epochs", help="number of epochs (default: 20)", type=int, default=30)
	parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=1024)
	parser.add_argument("--cuda", help="use CUDA", action="store_true")
	args = parser.parse_args()

	concentrations = [1e-2, 1e-1, 1.0, 10, 100]
	random_seeds = list(range(10))
	is_imb = args.imbalance

	f_train = open("./plans.txt", 'r')
	lines = f_train.readlines()
	x = []
	y = []
	for line in lines:
		line = line.strip()
		parts = line.split('|')
		y_i = parts[-1]
		x_i = "".join(parts[:len(parts) - 1])

		x.append(x_i)
		y.append(float(y_i))

	num_burnin = 2
	num_tasks = num_burnin + 3

	for random_seed in random_seeds:
		x_f, y_f, train_list, test_list = get_query_list(x, y, num_train_per_task=200, num_test_per_task=100, is_imb=is_imb,
		                                                 num_burnin=num_burnin,
		                                                 seed=random_seed)

		train_and_predict(x_f, y_f, train_list, test_list, args.cuda, 'rs',
		                  args.buffersize, num_tasks=num_tasks, seed=random_seed)

		train_and_predict(x_f, y_f, train_list, test_list, args.cuda, 'latest',
		                  args.buffersize, num_tasks=num_tasks, seed=random_seed)

		train_and_predict(x_f, y_f, train_list, test_list, args.cuda, 'all',
		                  args.buffersize, num_tasks=num_tasks, seed=random_seed)

		for concentration in concentrations:
			train_and_predict(x_f, y_f, train_list, test_list, args.cuda, 'cbp',
			                  args.buffersize, concentration=concentration, num_tasks=num_tasks, seed=random_seed)
			train_and_predict(x_f, y_f, train_list, test_list, args.cuda, 'lwp',
			                  args.buffersize, concentration=concentration, num_tasks=num_tasks, seed=random_seed)


if __name__ == "__main__":
	main()
