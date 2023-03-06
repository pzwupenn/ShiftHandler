import argparse
import time
import numpy as np
import torch
import json
from torch.autograd import Variable
from torch.utils.data import DataLoader
import random
from mscn.util import *
from mscn.data import get_workloads, get_task, get_task_test_queries
from mscn.model import SetConv
import sys

sys.path.append('../ShiftHandler')
from replay_buffer import summarizer


def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)


def qerror_loss(preds, targets, min_val, max_val, reduction=True):
    qerror = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    for i in range(len(targets)):
        if (preds[i] > targets[i]).cpu().data.numpy()[0]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])

    if reduction:
        return torch.mean(torch.cat(qerror))
    else:
        return torch.cat(qerror)


def predict(model, data_loader, cuda, min_val, max_val):
    preds = []
    labels = []
    t_total = 0.

    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):

        samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch
        labels.extend(targets.float())
        if cuda:
            samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
            sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
        samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(
            targets)
        sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
            join_masks)

        t = time.time()
        outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
        t_total += time.time() - t

        for i in range(outputs.data.shape[0]):
            preds.append(np.squeeze(outputs.data[i].cpu().detach().numpy()))

    preds_test_unnorm = unnormalize_labels(preds, min_val, max_val)
    labels_test_unnorm = unnormalize_labels(labels, min_val, max_val)

    qerrors = get_qerrors(preds_test_unnorm, labels_test_unnorm)
    return preds, t_total, qerrors

def predict_pg(pg_results, id_list, data_loader, cuda, min_val, max_val):
    preds = []
    labels = []
    t_total = 0.

    for batch_idx, data_batch in enumerate(data_loader):
        samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch
        labels.extend(targets.float())

    for i in range(len(labels)):
        preds.append(pg_results[id_list[i]])

    preds_test_unnorm = unnormalize_labels(preds, min_val, max_val)
    labels_test_unnorm = unnormalize_labels(labels, min_val, max_val)

    qerrors = get_qerrors(preds_test_unnorm, labels_test_unnorm)
    return preds, t_total, qerrors

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

def train_and_predict(num_queries, num_epochs, batch_size, hid_units, cuda, buffer, buffer_size, is_imbalance=False,
                      num_train=3600, num_test=400, num_tasks=5, num_burnin=4, concentration=1., tradeoff=0.5, seed=0):
    # Load training and validation data
    print("buffer: {}".format(buffer))
    print("seed: {}".format(seed))
    print("concentration: {}".format(concentration))

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    num_materialized_samples = 1000
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data, test_id_list = get_workloads(
        num_queries, num_materialized_samples, is_imbalance=is_imbalance, num_train=num_train, num_test=num_test, num_tasks=num_tasks,num_burnin=num_burnin, seed=seed)
    table2vec, column2vec, op2vec, join2vec = dicts

    # Train model
    sample_feats = len(table2vec) + num_materialized_samples
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = len(join2vec)

    model = SetConv(sample_feats, predicate_feats, join_feats, hid_units)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if cuda:
        model.cuda()

    prev_performs = []
    best_performs = []

    latest_buffer = []

    if buffer == 'lwp':
        HBRS_buffer = summarizer(buffer_limit=buffer_size, loss_ada=True,
                                 concentration=concentration,
                                 is_move=False)
    else:
        HBRS_buffer = summarizer(buffer_limit=buffer_size, loss_ada=False,
                                 concentration=concentration,
                                 is_move=False)

    num_queries_seen_far = 0
    num_tmp = num_tasks
    num_tasks = num_tasks + num_burnin

    for task_id in range(num_tasks):
        train_data_loader, test_data_loader = get_task(task_id, batch_size, train_data, test_data,
                                                       num_queries_per_workload=num_train,
                                                       num_test_queries_per_workload=num_test, seed=seed)

        avg_loss = 0.
        replay_losses_list = []
        current_losses_list = []

        model.train()
        for epoch in range(num_epochs):
            loss_total = 0.
            current_losses_list = []

            # first replay old queries
            replay_list = []
            if buffer == 'lwp' or buffer.lower() == 'cbp':
                replay_queries_tmp, _ = HBRS_buffer.get_all_samples()
                for (_, _, a_train_data, _, _) in replay_queries_tmp:
                    replay_list.append(a_train_data)
            elif buffer == 'latest':
                replay_list = latest_buffer
            elif buffer == 'rs':
                replay_list = latest_buffer
            elif buffer == 'all':
                replay_list = latest_buffer

            for batch_idx, data_batch in enumerate(train_data_loader):
                optimizer.zero_grad()
                samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch

                if cuda:
                    samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
                    sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
                samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(
                    joins), Variable(
                    targets)
                sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
                    join_masks)
                outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
                current_losses = qerror_loss(outputs, targets.float(), min_val, max_val, reduction=False)
                current_loss = torch.mean(current_losses)

                current_losses_list.extend(current_losses.cpu().detach().numpy())

                if len(replay_list):
                    # get the loss on the replay buffer
                    samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = [], [], [], [], [], [], []
                    for replay_data in replay_list:
                        samples.append(replay_data[0])
                        predicates.append(replay_data[1])
                        joins.append(replay_data[2])
                        targets.append(replay_data[3])
                        sample_masks.append(replay_data[4])
                        predicate_masks.append(replay_data[5])
                        join_masks.append(replay_data[6])

                    samples = np.expand_dims(samples, 0)
                    predicates = np.expand_dims(predicates, 0)
                    joins = np.expand_dims(joins, 0)
                    sample_masks = np.expand_dims(sample_masks, 0)
                    predicate_masks = np.expand_dims(predicate_masks, 0)
                    join_masks = np.expand_dims(join_masks, 0)

                    samples = np.vstack(samples)
                    samples = torch.FloatTensor(samples)
                    predicates = np.vstack(predicates)
                    predicates = torch.FloatTensor(predicates)
                    joins = np.vstack(joins)
                    joins = torch.FloatTensor(joins)
                    targets = np.vstack(targets)
                    targets = torch.FloatTensor(targets)
                    sample_masks = np.vstack(sample_masks)
                    sample_masks = torch.FloatTensor(sample_masks)
                    predicate_masks = np.vstack(predicate_masks)
                    predicate_masks = torch.FloatTensor(predicate_masks)
                    join_masks = np.vstack(join_masks)
                    join_masks = torch.FloatTensor(join_masks)

                    if cuda:
                        samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
                        sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()

                    samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(
                        joins), Variable(
                        targets)
                    sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(
                        predicate_masks), Variable(
                        join_masks)

                    replay_outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
                    replay_losses = qerror_loss(replay_outputs, targets.float(), min_val, max_val, reduction=False)

                    replay_loss = torch.mean(replay_losses)
                    loss = tradeoff * current_loss + (1 - tradeoff) * replay_loss

                    if buffer == 'lwp':
                        if epoch == num_epochs - 1 and batch_idx == len(train_data_loader) - 1:
                            replay_losses_list = replay_losses.cpu().detach().numpy()
                            HBRS_buffer.update_losses(replay_losses_list)

                else:
                    loss = current_loss
                loss_total += loss.item()
                loss.backward()
                optimizer.step()

        # print("Epoch {}, loss: {}".format(epoch, loss_total / len(train_data_loader)))

        if task_id > 3:
            preds_test, t_total, qerrors = predict(model, train_data_loader, cuda, min_val, max_val)
            res = {"buffer": buffer, "size": buffer_size, "concentration": concentration, "seed": seed,
                   "median": np.percentile(qerrors, 50), "95": np.percentile(qerrors, 95),
                   "max": np.max(qerrors), "mean": np.mean(qerrors)}
            best_performs.append(res)

            prev_perform_this_task = []
            for prev_task in range(num_tmp):
                test_queres_per_task = get_task_test_queries(prev_task, batch_size, test_data)
                preds_test, t_total, qerrors = predict(model, test_queres_per_task, cuda, min_val, max_val)
                res = {"buffer": buffer, "size": buffer_size, "concentration": concentration, "seed": seed,
                       "median": np.percentile(qerrors, 50), "95": np.percentile(qerrors, 95),
                       "max": np.max(qerrors), "mean": np.mean(qerrors)}
                prev_perform_this_task.append(res)

            prev_performs.append(prev_perform_this_task)

        if task_id == num_tasks - 1:
            break

        # add new queries to the replay buffer
        start_id = (task_id) * num_train
        end_id = (task_id + 1) * num_train
        all_train_data_loader = torch.utils.data.DataLoader(
            torch.utils.data.dataset.Subset(train_data, list(range(start_id, end_id))), batch_size=num_train)

        for _, data_batch in enumerate(all_train_data_loader):
            samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch
            targets = targets.cpu().detach().numpy()
            if buffer == 'lwp' or buffer.lower() == 'cbp':
                q_features = model.get_features(samples.cpu().detach().numpy(), predicates.cpu().detach().numpy(),
                                                joins.cpu().detach().numpy(), sample_masks.cpu().detach().numpy(),
                                                predicate_masks.cpu().detach().numpy(),
                                                join_masks.cpu().detach().numpy())
                for i in range(q_features.shape[0]):
                    q_feature = q_features[i, :]
                    original_data = (
                    samples[i, :, :].cpu().detach().numpy(), predicates[i, :, :].cpu().detach().numpy(),
                    joins[i, :, :].cpu().detach().numpy(), targets[i],
                    sample_masks[i, :, :].cpu().detach().numpy(), predicate_masks[i, :, :].cpu().detach().numpy(),
                    join_masks[i, :, :].cpu().detach().numpy())
                    if buffer == 'lwp':
                        HBRS_buffer.process_a_query(q_feature, targets[i], original_data, None, current_losses_list[i])
                    else:
                        HBRS_buffer.process_a_query(q_feature, targets[i], original_data)
            elif buffer == 'latest':
                for i in range(samples.shape[0]):
                    original_data = (
                    samples[i, :, :].cpu().detach().numpy(), predicates[i, :, :].cpu().detach().numpy(),
                    joins[i, :, :].cpu().detach().numpy(), targets[i],
                    sample_masks[i, :, :].cpu().detach().numpy(), predicate_masks[i, :, :].cpu().detach().numpy(),
                    join_masks[i, :, :].cpu().detach().numpy())
                    if len(latest_buffer) < buffer_size:
                        latest_buffer.append(original_data)
                    else:
                        latest_buffer.pop(0)
                        latest_buffer.append(original_data)
            elif buffer == 'rs':
                for i in range(samples.shape[0]):
                    original_data = (
                        samples[i, :, :].cpu().detach().numpy(), predicates[i, :, :].cpu().detach().numpy(),
                        joins[i, :, :].cpu().detach().numpy(), targets[i],
                        sample_masks[i, :, :].cpu().detach().numpy(), predicate_masks[i, :, :].cpu().detach().numpy(),
                        join_masks[i, :, :].cpu().detach().numpy())

                    if len(latest_buffer) < buffer_size:
                        latest_buffer.append(original_data)
                    else:
                        i = random.uniform(0, 1)
                        if i < float(len(latest_buffer)) / (num_queries_seen_far + 1):
                            latest_buffer.pop(0)
                            latest_buffer.append(original_data)
                    num_queries_seen_far += 1
            elif buffer == 'all':
                for i in range(samples.shape[0]):
                    original_data = (
                        samples[i, :, :].cpu().detach().numpy(), predicates[i, :, :].cpu().detach().numpy(),
                        joins[i, :, :].cpu().detach().numpy(), targets[i],
                        sample_masks[i, :, :].cpu().detach().numpy(), predicate_masks[i, :, :].cpu().detach().numpy(),
                        join_masks[i, :, :].cpu().detach().numpy())
                    latest_buffer.append(original_data)

    print("best_performs")
    print(best_performs)
    print("prev_performs")
    print(prev_performs)

    file_name = "./result_card_imb_{}".format(str(is_imbalance))
    with open(file_name, "a") as f:
        f.write("best\n")
        f.write("{}\n".format(json.dumps(best_performs)))
        f.write("prev\n")
        f.write("{}\n".format(json.dumps(prev_performs)))
        f.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imbalance', default=False, help="is imbalance?", action='store_true')
    parser.add_argument("--buffersize", help="buffer size", type=int, default=300)
    parser.add_argument("--queries", help="total number of queries (default: 10000)", type=int, default=100000)

    parser.add_argument("--numtrain", help="number of training queries per workload", type=int, default=1000)
    parser.add_argument("--numtest", help="number of test queries per workload", type=int, default=100)

    parser.add_argument("--epochs", help="number of epochs (default: 20)", type=int, default=30)
    parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=1000)
    parser.add_argument("--hid", help="number of hidden units (default: 256)", type=int, default=256)
    parser.add_argument("--cuda", help="use CUDA", action="store_true")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    is_imb = args.imbalance
    print('use_cuda')
    print(use_cuda)

    concentrations = [0.01, 0.1, 1, 10 ,100]

    random_seeds = list(range(10))

    for random_seed in random_seeds:
        train_and_predict(args.queries, args.epochs, args.batch, args.hid, use_cuda, 'all',
                          args.buffersize, is_imb, args.numtrain, args.numtest, concentration=1,
                          num_tasks=3, num_burnin=4, seed=random_seed)
        train_and_predict(args.queries, args.epochs, args.batch, args.hid, use_cuda, 'rs',
                          args.buffersize,is_imb, args.numtrain, args.numtest, concentration=1,
                          num_tasks=3, num_burnin=4, seed=random_seed)
        train_and_predict(args.queries, args.epochs, args.batch, args.hid, use_cuda, 'latest',
                          args.buffersize, is_imb, args.numtrain, args.numtest, concentration=1,
                          num_tasks=3, num_burnin=4, seed=random_seed)
        for concentration in concentrations:
            train_and_predict(args.queries, args.epochs, args.batch, args.hid, use_cuda, 'lwp',
                              args.buffersize, is_imb, args.numtrain, args.numtest, concentration=concentration,
                              num_tasks=3, num_burnin=4, seed=random_seed)
            train_and_predict(args.queries, args.epochs, args.batch, args.hid, use_cuda, 'cbp',
                              args.buffersize, is_imb, args.numtrain, args.numtest, concentration=concentration,
                              num_tasks=3, num_burnin=4, seed=random_seed)
if __name__ == "__main__":
    main()
