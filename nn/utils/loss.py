import torch
import torch.nn.functional as F

def physics_function_loss(X, Y, Z, u):
    laplacian = 0
    for j in range(8):
        u_i = u[:, j, :]
        
        grad_x = torch.autograd.grad(u_i, X, grad_outputs=torch.ones_like(u_i),
                                    create_graph=True, retain_graph=True)[0]
        grad_y = torch.autograd.grad(u_i, Y, grad_outputs=torch.ones_like(u_i),
                                    create_graph=True, retain_graph=True)[0]
        grad_z = torch.autograd.grad(u_i, Z, grad_outputs=torch.ones_like(u_i),
                                    create_graph=True, retain_graph=True)[0]
        grad2_x = torch.autograd.grad(grad_x, X, grad_outputs=torch.ones_like(grad_x),
                                    create_graph=True, retain_graph=True)[0]
        grad2_y = torch.autograd.grad(grad_y, Y, grad_outputs=torch.ones_like(grad_y),
                                    create_graph=True, retain_graph=True)[0]
        grad2_z = torch.autograd.grad(grad_z, Z, grad_outputs=torch.ones_like(grad_z),
                                    create_graph=True, retain_graph=True)[0]
        laplacian = grad2_x + grad2_y + grad2_z + laplacian
    laplacian = laplacian / 8

    return torch.mean(torch.abs(laplacian))

class ImprovedLoss:
    def __init__(self, peak_weight=2.0, gradient_weight=0.3, spectral_weight=0.2, 
                 constraint_weight=5.0, non_negative_weight=2.0):
        self.peak_weight = peak_weight
        self.gradient_weight = gradient_weight
        self.spectral_weight = spectral_weight
        self.constraint_weight = constraint_weight
        self.non_negative_weight = non_negative_weight
    
    def __call__(self, y_pred, y_true):
        y_pred_reshaped = y_pred.view(-1, 8, 125)
        y_true_reshaped = y_true.view(-1, 8, 125)
        
        mse_loss = F.mse_loss(y_pred, y_true)
        
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)
        
        peak_threshold = torch.quantile(y_true_flat, 0.75)
        peak_mask = y_true_flat > peak_threshold
        
        if peak_mask.sum() > 0:
            peak_loss = F.mse_loss(y_pred_flat[peak_mask], y_true_flat[peak_mask]) * self.peak_weight
        else:
            peak_loss = torch.tensor(0.0, device=y_pred.device)
        
        grad_pred = torch.diff(y_pred_reshaped, dim=-1)
        grad_true = torch.diff(y_true_reshaped, dim=-1)
        gradient_loss = F.mse_loss(grad_pred, grad_true) * self.gradient_weight
        
        try:
            fft_pred = torch.fft.fft(y_pred_reshaped, dim=-1)
            fft_true = torch.fft.fft(y_true_reshaped, dim=-1)
            spectral_loss = F.mse_loss(torch.abs(fft_pred), torch.abs(fft_true)) * self.spectral_weight
        except:
            spectral_loss = torch.tensor(0.0, device=y_pred.device)
        
        physics_constraint_loss = torch.tensor(0.0, device=y_pred.device)
        
        sum_constraint = torch.sum(y_pred_reshaped, dim=1)  # (batch, 125)
        target_sum = torch.ones_like(sum_constraint)
        normalization_loss = F.mse_loss(sum_constraint, target_sum) * self.constraint_weight
        
        negative_penalty = torch.sum(F.relu(-y_pred_reshaped)) * self.non_negative_weight
        
        upper_bound_penalty = torch.sum(F.relu(y_pred_reshaped - 1.0)) * self.non_negative_weight
        
        physics_constraint_loss = normalization_loss + negative_penalty + upper_bound_penalty
        
        total_loss = mse_loss + peak_loss + gradient_loss + spectral_loss + physics_constraint_loss
        
        return total_loss, {
            'mse': mse_loss.item(),
            'peak': peak_loss.item() if isinstance(peak_loss, torch.Tensor) else peak_loss,
            'gradient': gradient_loss.item(),
            'spectral': spectral_loss.item() if isinstance(spectral_loss, torch.Tensor) else spectral_loss,
            'normalization': normalization_loss.item(),
            'negative_penalty': negative_penalty.item(),
            'upper_bound_penalty': upper_bound_penalty.item(),
            'physics_total': physics_constraint_loss.item()
        }