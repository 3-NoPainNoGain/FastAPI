import torch
import torch.nn as nn

class SignLanguageBiLSTM(nn.Module):
    def __init__(self, input_size=258, hidden_size=256, num_layers=3, num_classes=11, dropout=0.5):
        super(SignLanguageBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)        
        out = out[:, -1, :]           
        out = self.bn(out)
        out = self.dropout(out)
        return self.fc(out)           
