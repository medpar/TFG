# model.py
import torch
import torch.nn as nn
import config

class GaitLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, 
                 bidirectional, lstm_dropout, linear_dropout):
        super(GaitLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # Input and output tensors are provided as (batch, seq, feature)
            bidirectional=bidirectional,
            dropout=lstm_dropout if num_layers > 1 else 0 # LSTM dropout only if num_layers > 1
        )
        
        lstm_output_features = hidden_size * 2 if bidirectional else hidden_size
        
        self.dropout = nn.Dropout(linear_dropout)
        self.fc = nn.Linear(lstm_output_features, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # Initialize hidden and cell states
        # h0 shape: (num_layers * num_directions, batch_size, hidden_size)
        # c0 shape: (num_layers * num_directions, batch_size, hidden_size)
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        # out shape: (batch_size, seq_len, hidden_size * num_directions)
        # hn shape: (num_layers * num_directions, batch_size, hidden_size)
        # cn shape: (num_layers * num_directions, batch_size, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Apply dropout
        out = self.dropout(out)
        
        # Pass LSTM output for each time step through the fully connected layer
        # out shape before fc: (batch_size, seq_len, hidden_size * num_directions)
        # We want predictions for each time step, so fc applies to the last dimension.
        # fc output shape: (batch_size, seq_len, num_classes)
        out = self.fc(out)
        
        return out # Logits for each timestep in the sequence

if __name__ == '__main__':
    print("Testing LSTM model definition...")
    # Example instantiation
    model = GaitLSTM(
        input_size=config.NUM_FEATURES,
        hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.NUM_LSTM_LAYERS,
        num_classes=config.NUM_CLASSES,
        bidirectional=config.BIDIRECTIONAL_LSTM,
        lstm_dropout=config.LSTM_DROPOUT,
        linear_dropout=config.LINEAR_DROPOUT
    ).to(config.DEVICE)
    
    print(model)
    
    # Test with a dummy batch
    batch_size = config.BATCH_SIZE
    seq_len = config.SEQUENCE_LENGTH
    dummy_input = torch.randn(batch_size, seq_len, config.NUM_FEATURES).to(config.DEVICE)
    output = model(dummy_input)
    print(f"\nDummy input shape: {dummy_input.shape}")
    print(f"Model output shape: {output.shape}") # Expected: [batch_size, seq_len, num_classes]