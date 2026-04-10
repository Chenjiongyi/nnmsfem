import torch.nn as nn
import torch
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.norm2(out)
        return self.activation(out + residual)

class PINN1(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(PINN1, self).__init__()
        self.input1_layer = nn.Sequential(
            nn.Linear(64, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            ResidualBlock(1024, 1024, dropout_rate),
            ResidualBlock(1024, 1024, dropout_rate),
        )

        self.input2_layer = nn.Sequential(
            nn.Linear(125*3, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            ResidualBlock(4096, 2048, dropout_rate),
            ResidualBlock(2048, 1024, dropout_rate),
        )

        self.shared_layer = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            
            ResidualBlock(2048, 4096, dropout_rate * 0.5),
            ResidualBlock(4096, 4096, dropout_rate * 0.5),
        )
        
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4096, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Dropout(dropout_rate * 0.3),
                nn.Linear(2048, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(dropout_rate * 0.2),
                nn.Linear(1024, 125),
            ) for _ in range(8)
        ])
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He初始化适用于GELU激活函数
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x_batch, X_tr, Y_tr, Z_tr):
        branch = self.input1_layer(x_batch)
        
        XYZ = torch.cat([X_tr, Y_tr, Z_tr], dim=1)
        
        trunk = self.input2_layer(XYZ)
        
        x = trunk * branch
        
        shared_features = self.shared_layer(x)
        
        outputs = []
        for head in self.output_heads:
            outputs.append(head(shared_features))
        
        output = torch.stack(outputs, dim=1)  # shape: (batch, 8, 125)
        
        # 应用物理约束：
        # 1. 使用ReLU确保非负性
        output = F.relu(output)
        
        # 2. 对每个空间点的8个基函数值进行softmax归一化，确保和为1
        # 重新整形为 (batch*125, 8) 进行softmax，然后再恢复形状
        batch_size = output.shape[0]
        output_reshaped = output.permute(0, 2, 1).contiguous().view(-1, 8)  # (batch*125, 8)
        output_normalized = F.softmax(output_reshaped, dim=1)  # 每个空间点的8个值归一化
        output_final = output_normalized.view(batch_size, 125, 8).permute(0, 2, 1)  # 恢复为 (batch, 8, 125)
        
        return output_final.reshape(-1, 8*125)