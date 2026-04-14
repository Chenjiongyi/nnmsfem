import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from utils.model import PINN1
from utils.dataset import get_data

plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PINN1().to(device)
ckpt_path = 'results/model_best.ckpt'
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

data_dir = 'data/test/'
x1, x2, y = get_data(data_dir, device)

num_samples = x1.shape[0]
x1_sample = x1[[0]]
x2_sample = x2[[0]]
y_true_sample = y[[0]]

def predict(model, x1, x2):
    X_arr = x2[:, :, 0]
    Y_arr = x2[:, :, 1]
    Z_arr = x2[:, :, 2]
    X_tr = X_arr.to(device)
    Y_tr = Y_arr.to(device)
    Z_tr = Z_arr.to(device)
    X_tr.requires_grad_(True)
    Y_tr.requires_grad_(True)
    Z_tr.requires_grad_(True)
    with torch.no_grad():
        y_pred = model(x1, X_tr, Y_tr, Z_tr)
        y_pred = y_pred.view(-1, 8, 125)
    return y_pred

y_pred_sample = predict(model, x1_sample, x2_sample)

plt.figure(figsize=(15, 5))
plt.plot(y_true_sample[0].cpu().numpy().flatten(), label='True')
plt.plot(y_pred_sample[0].cpu().numpy().flatten(), label='Pred', linestyle='--')
plt.legend()
plt.grid(True)
plt.xlabel('Data Points')
plt.ylabel('Values')
plt.tight_layout()
plt.savefig("result.png")