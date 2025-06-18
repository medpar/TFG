# model_30Hz/model_30Hz.py
import torch
import torch.nn as nn
import config_30Hz as config # Use the 30Hz config

class GaitLSTM30Hz(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, 
                 bidirectional, lstm_dropout, linear_dropout):
        super(GaitLSTM30Hz, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout if num_layers > 1 else 0
        )
        
        lstm_output_features = hidden_size * 2 if bidirectional else hidden_size
        
        self.dropout = nn.Dropout(linear_dropout)
        self.fc = nn.Linear(lstm_output_features, num_classes)

    def forward(self, x):
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

if __name__ == '__main__':
    print("Testing 30Hz LSTM model definition...")
    model = GaitLSTM30Hz(
        input_size=config.NUM_FEATURES,
        hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.NUM_LSTM_LAYERS,
        num_classes=config.NUM_CLASSES,
        bidirectional=config.BIDIRECTIONAL_LSTM,
        lstm_dropout=config.LSTM_DROPOUT,
        linear_dropout=config.LINEAR_DROPOUT
    ).to(config.DEVICE)
    print(model)