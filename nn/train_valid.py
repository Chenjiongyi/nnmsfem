import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from datetime import datetime
import os

from utils.dataset import get_data, DataAugmentation
from utils.loss import physics_function_loss, ImprovedLoss
from utils.model import PINN1 as PINN

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * epoch / self.warmup_epochs
        else:
            lr = self.eta_min + (self.base_lr - self.eta_min) * 0.5 * \
                 (1 + np.cos(np.pi * (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

def train_model(config, model, train_loader, valid_loader, device, fold=0):
    start_time = datetime.now()
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config["learning_rate"], 
                           weight_decay=config["weight_decay"])
    
    if config.get("use_cosine_scheduler", False):
        scheduler = CosineWarmupScheduler(optimizer, 
                                       config["warmup_epochs"], 
                                       config["epochs"])
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config["scheduler_factor"],
            patience=config["scheduler_patience"],
            min_lr=config["min_lr"]
        )
    train_loss_list = []
    valid_loss_list = []
    best_loss = float('inf')
    stale = 0
    last_lr = config["learning_rate"]
    
    improved_criterion = ImprovedLoss(
        peak_weight=config.get("peak_weight", 2.0),
        gradient_weight=config.get("gradient_weight", 0.3),
        spectral_weight=config.get("spectral_weight", 0.2),
        constraint_weight=config.get("constraint_weight", 5.0),
        non_negative_weight=config.get("non_negative_weight", 2.0)
    )
    
    def get_adaptive_weights(epoch, total_epochs):
        data_weight = max(0.8 - 0.3 * epoch / total_epochs, 0.3)
        physics_weight = min(0.2 + 0.5 * epoch / total_epochs, 0.7)
        return data_weight, physics_weight
    
    data_augmentor = DataAugmentation() if config.get("use_data_augmentation", False) else None
    
    history = {
        'train_loss': [], 'train_data_loss': [], 'train_physics_loss': [],
        'valid_loss': [], 'valid_data_loss': [], 'valid_physics_loss': [],
        'learning_rate': []
    }
    
    for epoch in range(config["epochs"]):
        model.train()
        total_train_loss = 0
        total_train_data_loss = 0
        total_train_physics_loss = 0
        
        current_data_weight, current_physics_weight = get_adaptive_weights(epoch, config["epochs"])
        
        for x1_batch, x2_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            if data_augmentor is not None:
                x1_batch, x2_batch, y_batch = data_augmentor(x1_batch, x2_batch, y_batch)
            
            X_arr, Y_arr, Z_arr = [], [], []
            for i in range(x2_batch.size()[0]):
                X_arr.append(x2_batch[i][:, 0].cpu().numpy())
                Y_arr.append(x2_batch[i][:, 1].cpu().numpy())
                Z_arr.append(x2_batch[i][:, 2].cpu().numpy())
            X_tr = torch.tensor(np.array(X_arr), dtype=torch.float32).to(device)
            Y_tr = torch.tensor(np.array(Y_arr), dtype=torch.float32).to(device)
            Z_tr = torch.tensor(np.array(Z_arr), dtype=torch.float32).to(device)
            X_tr.requires_grad_(True)
            Y_tr.requires_grad_(True)
            Z_tr.requires_grad_(True)
            y_pred = model(x1_batch, X_tr, Y_tr, Z_tr)
            y_pred = y_pred.view(-1, 8, 125)
            
            data_loss_value, loss_components = improved_criterion(y_pred, y_batch)
            physics_loss = physics_function_loss(X_tr, Y_tr, Z_tr, y_pred)
            
            loss = current_data_weight * data_loss_value + current_physics_weight * physics_loss
            
            if config.get("l2_regularization", 0) > 0:
                l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
                loss += config["l2_regularization"] * l2_reg
            
            loss.backward()
            
            if config.get("gradient_clipping", 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clipping"])
            
            optimizer.step()
            data_loss = data_loss_value
            total_train_loss += loss.item()
            total_train_data_loss += data_loss.item()
            total_train_physics_loss += physics_loss.item()

        model.eval()
        total_valid_loss = 0
        total_valid_data_loss = 0
        total_valid_physics_loss = 0
        
        for x1_batch, x2_batch, y_batch in valid_loader:
            X_arr, Y_arr, Z_arr = [], [], []
            for i in range(x2_batch.size()[0]):
                X_arr.append(x2_batch[i][:, 0].cpu().numpy())
                Y_arr.append(x2_batch[i][:, 1].cpu().numpy())
                Z_arr.append(x2_batch[i][:, 2].cpu().numpy())
            X_tr = torch.tensor(np.array(X_arr), dtype=torch.float32).to(device)
            Y_tr = torch.tensor(np.array(Y_arr), dtype=torch.float32).to(device)
            Z_tr = torch.tensor(np.array(Z_arr), dtype=torch.float32).to(device)
            X_tr.requires_grad_(True)
            Y_tr.requires_grad_(True)
            Z_tr.requires_grad_(True)
            
            y_pred = model(x1_batch, X_tr, Y_tr, Z_tr)
            y_pred = y_pred.view(-1, 8, 125)
            
            with torch.no_grad():
                data_loss_value, _ = improved_criterion(y_pred, y_batch)
            
            physics_loss = physics_function_loss(X_tr, Y_tr, Z_tr, y_pred)
            loss = current_data_weight * data_loss_value + current_physics_weight * physics_loss
            
            total_valid_loss += loss.item()
            total_valid_data_loss += data_loss_value.item()
            total_valid_physics_loss += physics_loss.item()
        
        val_loss = total_valid_loss/len(valid_loader)

        current_lr = optimizer.param_groups[0]['lr']
        
        if config.get("use_cosine_scheduler", False):
            new_lr = scheduler.step(epoch)
        else:
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']

        if new_lr != last_lr:
            print(f"Learning rate update from {last_lr:.2e} to {new_lr:.2e}")
            last_lr = new_lr

        train_loss = total_train_loss/len(train_loader)
        valid_loss = total_valid_loss/len(valid_loader)
        train_data_loss = total_train_data_loss/len(train_loader)
        train_physics_loss = total_train_physics_loss/len(train_loader)
        valid_data_loss = total_valid_data_loss/len(valid_loader)
        valid_physics_loss = total_valid_physics_loss/len(valid_loader)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        
        history['train_loss'].append(train_loss)
        history['train_data_loss'].append(train_data_loss)
        history['train_physics_loss'].append(train_physics_loss)
        history['valid_loss'].append(valid_loss)
        history['valid_data_loss'].append(valid_data_loss)
        history['valid_physics_loss'].append(valid_physics_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Valid Loss = {valid_loss:.6f}, '
              f'Data Weight = {current_data_weight:.3f}, Physics Weight = {current_physics_weight:.3f}, '
              f'Learning Rate = {current_lr:.2e}')
        
        if epoch == 0:
            header = "epoch train_loss train_data_loss train_physics_loss valid_loss valid_data_loss valid_physics_loss learning_rate"
            with open(f"{save_dir}/training_history.txt", "w") as f:
                f.write(header + "\n")
        
        with open(f"{save_dir}/training_history.txt", "a") as f:
            f.write(f"{epoch+1}\t{train_loss:.6e}\t{train_data_loss:.6e}\t{train_physics_loss:.6e}\t{valid_loss:.6e}\t{valid_data_loss:.6e}\t{valid_physics_loss:.6e}\t{current_lr:.6e}\n")
    
        if valid_loss < best_loss:
            print(f"Best model found at fold {fold+1} epoch {epoch+1}, Valid Loss={valid_loss}, saving model")
            torch.save(model.state_dict(), f"results/{config['start']}/fold_{fold+1}_best.ckpt")
            best_loss = valid_loss
            stale = 0
        else:
            stale += 1
            if stale > config["patience"]:
                print(f'No improvment {config["patience"]} consecutive epochs, early stopping')
                break
    
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"\nTotal training time: {training_time}")
    with open(f"{save_dir}/config.txt", "a") as f:
        f.write(f"training_time: {training_time}\n")
    
    return model, train_loss_list, valid_loss_list

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"results/{timestamp}"
os.makedirs(save_dir, exist_ok=True)

config = {
    "start": timestamp,
    "seed": 42,
    "dataset": "10000",
    "learning_rate": 5e-4,
    "epochs": 2000,
    "batch_size": 16,
    "patience": 150,
    "scheduler_patience": 30,
    "scheduler_factor": 0.7,
    "min_lr": 1e-8,
    
    "use_cosine_scheduler": True,
    "warmup_epochs": 50,
    "weight_decay": 1e-5,
    "gradient_clipping": 1.0,
    "l2_regularization": 1e-6,
    
    "peak_weight": 3.0,
    "gradient_weight": 0.5, 
    "spectral_weight": 0.3,
    "constraint_weight": 10.0,
    "non_negative_weight": 5.0,
    
    "use_data_augmentation": True,
}

with open(f"{save_dir}/config.txt", "w") as f:
    for key, value in config.items():
        f.write(f"{key}: {value}\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**2:.1f} MB")

torch.manual_seed(config["seed"])

url = 'data/' + config["dataset"] + '/'
x1, x2, y = get_data(url, device)

print("Data shape:")
print(f"x1: {x1.shape};")
print(f"x2: {x2.shape};")
print(f"y: {y.shape};")

data = TensorDataset(x1, x2, y)

train_indices, valid_indices = train_test_split(range(len(data)), test_size=0.2, random_state=config["seed"])
train_subset = Subset(data, train_indices)
valid_subset = Subset(data, valid_indices)

train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True)
valid_loader = DataLoader(valid_subset, batch_size=config["batch_size"], shuffle=False)

model = PINN().to(device)

trained_model, train_loss_list, valid_loss_list = train_model(config, model, train_loader, valid_loader, device)