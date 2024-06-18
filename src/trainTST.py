import random
import numpy as np
import os
import torch.nn as nn
import math
import torch
from load_data import load_EOD_data
from evaluator import evaluate
from model import get_loss
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:(d_model // 2)])
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerStockMixer(nn.Module):
    def __init__(self, input_dim, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(TransformerStockMixer, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(input_dim, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.encoder = nn.Linear(input_dim, input_dim)
        self.decoder = nn.Linear(input_dim, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(src.size(2))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output
    
def get_batch(offset=None):
    if offset is None:
        offset = random.randrange(0, valid_index)
    seq_len = lookback_length
    mask_batch = mask_data[:, offset: offset + seq_len + steps]
    mask_batch = np.min(mask_batch, axis=1)
    return (
        torch.Tensor(eod_data[:, offset:offset + seq_len, :]).to(device),
        torch.Tensor(np.expand_dims(mask_batch, axis=1)).to(device),
        torch.Tensor(np.expand_dims(price_data[:, offset + seq_len - 1], axis=1)).to(device),
        torch.Tensor(np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1)).to(device)
    )

def validate(model, start_index, end_index):
    model.eval()
    with torch.no_grad():
        cur_valid_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
        loss = 0.
        reg_loss = 0.
        rank_loss = 0.
        for cur_offset in range(start_index - lookback_length - steps + 1, end_index - lookback_length - steps + 1):
            data_batch, mask_batch, price_batch, gt_batch = get_batch(cur_offset)
            prediction = model(data_batch).squeeze(-1)
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
# Hyperparameters
input_dim = fea_num
nhead = 1
num_encoder_layers = 3
dim_feedforward = 128
dropout = 0.1

model = TransformerStockMixer(input_dim, nhead, num_encoder_layers, dim_feedforward, dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_valid_loss = np.inf
best_valid_perf = None
best_test_perf = None
batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)

for epoch in range(epochs):
    print("epoch{}##########################################################".format(epoch + 1))
    np.random.shuffle(batch_offsets)
    model.train()
    tra_loss = 0.0
    tra_reg_loss = 0.0
    tra_rank_loss = 0.0

    for j in range(valid_index - lookback_length - steps + 1):
        data_batch, mask_batch, price_batch, gt_batch = get_batch(batch_offsets[j])
        optimizer.zero_grad()
        prediction = model(data_batch).squeeze(-1)
        cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                            stock_num, alpha)
        cur_loss.backward()
        optimizer.step()

        tra_loss += cur_loss.item()
        tra_reg_loss += cur_reg_loss.item()
        tra_rank_loss += cur_rank_loss.item()
    tra_loss = tra_loss / (valid_index - lookback_length - steps + 1)
    tra_reg_loss = tra_reg_loss / (valid_index - lookback_length - steps + 1)
    tra_rank_loss = tra_rank_loss / (valid_index - lookback_length - steps + 1)
    print('Train : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(tra_loss, tra_reg_loss, tra_rank_loss))

    val_loss, val_reg_loss, val_rank_loss, val_perf = validate(model, valid_index, test_index)
    print('Valid : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(val_loss, val_reg_loss, val_rank_loss))

    test_loss, test_reg_loss, test_rank_loss, test_perf = validate(model, test_index, trade_dates)
    print('Test: loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(test_loss, test_reg_loss, test_rank_loss))

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        best_valid_perf = val_perf
        best_test_perf = test_perf

    print('Valid performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(val_perf['mse'], val_perf['IC'],
                                                         val_perf['RIC'], val_perf['prec_10'], val_perf['sharpe5']))
    print('Test performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(test_perf['mse'], test_perf['IC'],
                                                                        test_perf['RIC'], test_perf['prec_10'], test_perf['sharpe5']), '\n\n')
