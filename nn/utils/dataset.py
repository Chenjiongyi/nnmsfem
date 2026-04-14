import torch
import numpy as np

def get_data(url, device):
    localr_file = url + 'localr.txt'
    with open(localr_file) as f:
        lines = f.readlines()
    x1 = torch.tensor([list(map(float, line.strip().split(','))) for line in lines], dtype=torch.float32).to(device)
    x1 = torch.log10(x1)

    xyz_file = url + 'xyz.txt'
    with open(xyz_file) as f:
        lines = f.readlines()
    xyz = torch.tensor([list(map(float, line.strip().split(','))) for line in lines], dtype=torch.float32).to(device)
    dof_num = int(xyz.size(1) / 3)
    x2 = xyz.view(-1, dof_num, 3)

    solution_file = url + 'solution.txt'
    with open(solution_file) as f:
        lines = f.readlines()
    solutions = []
    for line in lines:
        values = [float(val) for val in line.strip().split(',')]
        solutions.append(values)
    solution = torch.tensor(solutions, dtype=torch.float32).to(device)
    y = solution.view(-1, 8, dof_num)

    print(x1.size(), x2.size(), y.size())
    return x1, x2, y

class DataAugmentation:
    def __init__(self, noise_std=0.01, coord_noise_std=0.001):
        self.noise_std = noise_std
        self.coord_noise_std = coord_noise_std
        
    def __call__(self, x1, x2, y):
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(y) * self.noise_std
            y = y + noise
            
        if torch.rand(1) < 0.2:
            scale = 0.98 + (1.02 - 0.98) * torch.rand(1, device=y.device)
            y = y * scale
        
        if torch.rand(1) < 0.3:
            coord_noise = torch.randn_like(x2) * self.coord_noise_std
            x2 = x2 + coord_noise
            
        return x1, x2, y