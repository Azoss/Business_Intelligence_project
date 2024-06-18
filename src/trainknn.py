import random
import numpy as np
import os
import torch as torch
from load_data import load_EOD_data
from evaluator import evaluate
from model import get_loss, StockMixer, KNNStockMixer
import pickle



np.random.seed(123456789)
torch.random.manual_seed(12345678)
device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

# Load data
data_path = '../dataset'
market_name = 'NASDAQ'
relation_name = 'wikidata'
stock_num = 1026
lookback_length = 16
epochs = 15
valid_index = 756
test_index = 1008
fea_num = 5
market_num = 20
steps = 1
learning_rate = 0.001
alpha = 0.1
scale_factor = 3
activation = 'GELU'

dataset_path = '../dataset/' + market_name
with open(os.path.join(dataset_path, "/media/isk/New Volume/Kuliah/Semester_6/Business Intelligence/Code/Business_Intelligence_project/dataset/NASDAQ/eod_data.pkl"), "rb") as f:
        eod_data = pickle.load(f)
with open(os.path.join(dataset_path, "/media/isk/New Volume/Kuliah/Semester_6/Business Intelligence/Code/Business_Intelligence_project/dataset/NASDAQ/mask_data.pkl"), "rb") as f:
    mask_data = pickle.load(f)
with open(os.path.join(dataset_path, "/media/isk/New Volume/Kuliah/Semester_6/Business Intelligence/Code/Business_Intelligence_project/dataset/NASDAQ/gt_data.pkl"), "rb") as f:
    gt_data = pickle.load(f)
with open(os.path.join(dataset_path, "/media/isk/New Volume/Kuliah/Semester_6/Business Intelligence/Code/Business_Intelligence_project/dataset/NASDAQ/price_data.pkl"), "rb") as f:
    price_data = pickle.load(f)

trade_dates = mask_data.shape[1]
model = KNNStockMixer(n_neighbors=5)

def get_batch(offset=None):
    if offset is None:
        offset = random.randrange(0, valid_index)
    seq_len = lookback_length
    mask_batch = mask_data[:, offset: offset + seq_len + steps]
    mask_batch = np.min(mask_batch, axis=1)
    return (
        eod_data[:, offset:offset + seq_len, :],
        np.expand_dims(mask_batch, axis=1),
        np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
        np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1))

def validate(start_index, end_index):
    with torch.no_grad():
        cur_valid_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
        loss = 0.
        reg_loss = 0.
        rank_loss = 0.
        for cur_offset in range(start_index - lookback_length - steps + 1, end_index - lookback_length - steps + 1):
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device),
                get_batch(cur_offset)
            )
            prediction = model.predict(data_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                     stock_num, alpha)
            loss += cur_loss.item()
            reg_loss += cur_reg_loss.item()
            rank_loss += cur_rank_loss.item()
            cur_valid_pred[:, cur_offset - (start_index - lookback_length - steps + 1)] = cur_rr[:, 0].cpu()
            cur_valid_gt[:, cur_offset - (start_index - lookback_length - steps + 1)] = gt_batch[:, 0].cpu()
            cur_valid_mask[:, cur_offset - (start_index - lookback_length - steps + 1)] = mask_batch[:, 0].cpu()
        loss = loss / (end_index - start_index)
        reg_loss = reg_loss / (end_index - start_index)
        rank_loss = rank_loss / (end_index - start_index)
        cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
    return loss, reg_loss, rank_loss, cur_valid_perf

# Training
best_valid_loss = float('inf')
best_valid_perf = None
best_test_perf = None

for epoch in range(epochs):
    print("epoch{}##########################################################".format(epoch + 1))
    batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)
    np.random.shuffle(batch_offsets)
    tra_loss = 0.0
    tra_reg_loss = 0.0
    tra_rank_loss = 0.0

    # Prepare the training data for fitting KNN
    train_data = []
    train_gt = []

    for j in range(valid_index - lookback_length - steps + 1):
        data_batch, mask_batch, price_batch, gt_batch = map(
            lambda x: torch.Tensor(x).to(device),
            get_batch(batch_offsets[j])
        )
        train_data.append(data_batch)
        train_gt.append(gt_batch)

    train_data = torch.cat(train_data, dim=0)
    train_gt = torch.cat(train_gt, dim=0)

    model.fit(train_data, train_gt)

    val_loss, val_reg_loss, val_rank_loss, val_perf = validate(valid_index, test_index)
    print('Valid : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(val_loss, val_reg_loss, val_rank_loss))

    test_loss, test_reg_loss, test_rank_loss, test_perf = validate(test_index, trade_dates)
    print('Test: loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(test_loss, test_reg_loss, test_rank_loss))

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        best_valid_perf = val_perf
        best_test_perf = test_perf

    print('Valid performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(val_perf['mse'], val_perf['IC'],
                                                     val_perf['RIC'], val_perf['prec_10'], val_perf['sharpe5']))
    print('Test performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(test_perf['mse'], test_perf['IC'],
                                                                            test_perf['RIC'], test_perf['prec_10'], test_perf['sharpe5']), '\n\n')

print('Best Valid performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
    best_valid_perf['mse'], best_valid_perf['IC'], best_valid_perf['RIC'], best_valid_perf['prec_10'], best_valid_perf['sharpe5']))
print('Best Test performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
    best_test_perf['mse'], best_test_perf['IC'], best_test_perf['RIC'], best_test_perf['prec_10'], best_test_perf['sharpe5']))