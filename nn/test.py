import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from utils.model import PINN1
from utils.dataset import get_data

plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PINN1().to(device)
ckpt_path = 'results/fold_1_best.ckpt'
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

data_dir = 'data/model2/'
x1, x2, y = get_data(data_dir, device)

num_samples = x1.shape[0]
indices = np.random.choice(num_samples, 10, replace=False)
x1_sample = x1[indices]
x2_sample = x2[indices]
y_true_sample = y[indices]

def predict(model, x1, x2):
    X_arr = x2[:, 0]  # [125]
    Y_arr = x2[:, 1]  # [125]
    Z_arr = x2[:, 2]  # [125]
    X_tr = X_arr.unsqueeze(0).to(device)  # [1, 125]
    Y_tr = Y_arr.unsqueeze(0).to(device)  # [1, 125]
    Z_tr = Z_arr.unsqueeze(0).to(device)  # [1, 125]
    X_tr.requires_grad_(True)
    Y_tr.requires_grad_(True)
    Z_tr.requires_grad_(True)
    with torch.no_grad():
        y_pred = model(x1.unsqueeze(0), X_tr, Y_tr, Z_tr)
        y_pred = y_pred.view(8, 125)
    return y_pred

y_pred_sample = []
for i in range(10):
    y_pred = predict(model, x1_sample[i], x2_sample[i])
    y_pred_sample.append(y_pred)
y_pred_sample = torch.stack(y_pred_sample)

fig, axes = plt.subplots(10, 1, figsize=(15, 30))
for i in range(10):
    axes[i].plot(y_true_sample[i].cpu().numpy().flatten(), label='True')
    axes[i].plot(y_pred_sample[i].cpu().numpy().flatten(), label='Pred', linestyle='--')
    axes[i].set_title(f'Sample {indices[i]}')
    axes[i].legend()
plt.tight_layout()
plt.show()

