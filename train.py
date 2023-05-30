import os
import time
import argparse
import numpy as np
import torch
from model import EDITS
from utils import load_bail, load_credit, load_german,load_pokec, sparse_mx_to_torch_sparse_tensor, normalize_scipy, feature_norm
import scipy.sparse as sp
from tqdm import tqdm
from metrics import metric_wd
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import time
from memory_profiler import memory_usage

# Training settings
start_time = time.time()
mem_usage = memory_usage()[0]

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--cuda_device', type=int, default=0,
                    help='cuda device running on.')
parser.add_argument('--dataset', type=str, default='bail',
                    help='a dataset from credit, german and bail.')
parser.add_argument('--epochs', type=int, default=100,  # bail: 100; others: 500.
                    help='Number of epochs to train.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.003,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-7,
                    help='Weight decay (L2 loss on parameters).')
args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def binarize(A_debiased, adj_ori, threshold_proportion):

    the_con1 = (A_debiased - adj_ori).A
    the_con1 = np.where(the_con1 > np.max(the_con1) * threshold_proportion, 1 + the_con1 * 0, the_con1)
    the_con1 = np.where(the_con1 < np.min(the_con1) * threshold_proportion, -1 + the_con1 * 0, the_con1)
    the_con1 = np.where(np.abs(the_con1) == 1, the_con1, the_con1 * 0)
    A_debiased = adj_ori + sp.coo_matrix(the_con1)
    assert A_debiased.max() == 1
    assert A_debiased.min() == 0
    A_debiased = normalize_scipy(A_debiased)
    return A_debiased


if args.dataset == 'credit':
    sens_idx = 1
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit('credit', label_number=6000,seed=args.seed)
elif args.dataset == 'german':
    sens_idx = 0
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_german('german', label_number=100,seed=args.seed)
elif args.dataset == 'bail':
    sens_idx = 0
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail('bail', label_number=100,seed=args.seed)

elif args.dataset == 'pokec_z':
    dataset = 'region_job'
    sens_attr = "region"
    predict_attr = "I_am_working_in_field"
    label_number = 500
    sens_number = 200
    seed = args.seed
    path = "./dataset/pokec/"
    test_idx = False
    sens_idx = 3
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_pokec(dataset,
                                                                           sens_attr,
                                                                           predict_attr,
                                                                           path=path,
                                                                           label_number=label_number,
                                                                           sens_number=sens_number,
                                                                           seed=seed, test_idx=test_idx)
# 需要做norm吗？
# Load pokec dataset
elif args.dataset == 'pokec_n':
    dataset = 'region_job_2'
    sens_attr = "region"
    predict_attr = "I_am_working_in_field"
    label_number = 500
    sens_number = 200
    seed = args.seed
    path = "./dataset/pokec/"
    test_idx = False
    sens_idx = 3
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_pokec(dataset,
                                                                           sens_attr,
                                                                           predict_attr,
                                                                           path=path,
                                                                           label_number=label_number,
                                                                           sens_number=sens_number,
                                                                           seed=seed, test_idx=test_idx)
else:
    print("This dataset is not supported up to now!")

print("****************************Before debiasing****************************")
if args.dataset != 'german':
    preserve = features
    norm_features = feature_norm(features)
    norm_features[:, sens_idx] = features[:, sens_idx]
    # features = norm_features
else:
    norm_features = features
# metric_wd(norm_features, normalize_scipy(adj), sens, 0.9, 0)
# metric_wd(norm_features, normalize_scipy(adj), sens, 0.9, 2)
print("****************************************************************************")

features_preserve = features.clone()
features = features / features.norm(dim=0)
adj_preserve = adj
adj = sparse_mx_to_torch_sparse_tensor(adj)
model = EDITS(args, nfeat=features.shape[1], node_num=features.shape[0], nfeat_out=int(features.shape[0]/10), adj_lambda=1e-1, nclass=2, layer_threshold=2, dropout=0.2)  # 3-nba

if args.cuda:
    torch.cuda.set_device(args.cuda_device)
    model.cuda().half()
    adj = adj.cuda().half()
    features = features.cuda().half()
    labels = labels.cuda().half()
    features_preserve = features_preserve.cuda().half()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    sens = sens.cuda()


A_debiased, X_debiased = adj, features
val_adv = []
test_adv = []
for epoch in tqdm(range(args.epochs)):
    if epoch > 400:
        args.lr = 0.001
    model.train()
    model.optimize(adj, features, idx_train, sens, epoch, args.lr)
    A_debiased, X_debiased, predictor_sens, show, _ = model(adj, features)
    positive_eles = torch.masked_select(predictor_sens[idx_val].squeeze(), sens[idx_val] > 0)
    negative_eles = torch.masked_select(predictor_sens[idx_val].squeeze(), sens[idx_val] <= 0)
    loss_val = - (torch.mean(positive_eles) - torch.mean(negative_eles))
    val_adv.append(loss_val.data)

param = model.state_dict()

indices = torch.argsort(param["x_debaising.s"])[:4]
for i in indices:
    features_preserve[:, i] = torch.zeros_like(features_preserve[:, i])
X_debiased = features_preserve
adj1 = sp.csr_matrix(A_debiased.detach().cpu().numpy())

end_time = time.time()
max_mem_usage = max(memory_usage()) - mem_usage
run_time = end_time - start_time

print('run time {} s'.format(run_time))
print('max memory usage {} MB'.format(max_mem_usage))
# print("****************************After debiasing****************************")  # threshold_proportion for GCN: {credit: 0.02, german: 0.25, bail: 0.012}
# features1 = X_debiased.cpu().float()[:, torch.nonzero(features.sum(axis=0)).squeeze()].detach()
# if args.dataset != 'german':
#     features1 = feature_norm(features1)
# metric_wd(features1, binarize(adj1, adj_preserve, 0.012), sens.cpu(), 0.9, 0)
# metric_wd(features1, binarize(adj1, adj_preserve, 0.012), sens.cpu(), 0.9, 2)
# print("****************************************************************************")
sp.save_npz('pre_processed/A_debiased.npz', adj1)
torch.save(X_debiased, "pre_processed/X_debiased.pt")
print("Preprocessed datasets saved.")
