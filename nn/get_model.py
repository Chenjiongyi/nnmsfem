import torch
import torch.nn as nn
import torch.onnx
import numpy as np

from utils.model import PINN1

print(torch.__version__)
print(torch.version.cuda)
device = 'cpu'

model = PINN1().to(device)
model_url = 'results/fold_1_best.ckpt'
model.load_state_dict(torch.load(model_url))
print('Model loaded successfully!')

x1 = torch.randn(1, 64)
x2 = torch.randn(1, 125, 3)
X_arr = [x2[0, :, 0].cpu().numpy()]
Y_arr = [x2[0, :, 1].cpu().numpy()]
Z_arr = [x2[0, :, 2].cpu().numpy()]
X = torch.tensor(np.array(X_arr), dtype=torch.float32)
Y = torch.tensor(np.array(Y_arr), dtype=torch.float32)
Z = torch.tensor(np.array(Z_arr), dtype=torch.float32)

model.eval()

torch.onnx.export(
    model,
    args=(x1, X, Y, Z),
    f="results/model.onnx",       # 导出的ONNX文件名
    export_params=True,       # 是否导出模型参数
    opset_version=11,        # ONNX算子集的版本号
    input_names=['x1', 'X', 'Y', 'Z'],  # 输入张量的名称
    output_names=['output'],  # 输出张量的名称
)

print('Model saved as my_model.onnx')