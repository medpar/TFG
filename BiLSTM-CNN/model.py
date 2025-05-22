# model/model.py
import torch
import torch.nn as nn
# Removed config import from here, model should be self-contained
# Hyperparameters will be passed during instantiation

class BiLSTMCnn1D(nn.Module):
    def __init__(self, input_size, num_classes,
                 cnn_out_channels, cnn_kernel_sizes, cnn_strides, cnn_padding, cnn_activation, cnn_dropout_rate,
                 lstm_hidden_size, num_lstm_layers, lstm_dropout_rate, bidirectional_lstm,
                 linear_dropout_rate):
        super(BiLSTMCnn1D, self).__init__()

        if len(cnn_out_channels) != len(cnn_kernel_sizes) or len(cnn_out_channels) != len(cnn_strides):
            raise ValueError("CNN channels, kernels, and strides lists must have the same length.")

        # --- 1D CNN Layers ---
        cnn_layers = []
        current_in_channels = input_size

        for i in range(len(cnn_out_channels)):
            out_channels = cnn_out_channels[i]
            kernel_size = cnn_kernel_sizes[i]
            stride = cnn_strides[i]
            padding_val = 0 # Default

            if isinstance(cnn_padding, str) and cnn_padding.lower() == 'same':
                # For 'same' padding with stride 1, padding = (kernel_size - 1) // 2 (assuming dilation 1)
                # PyTorch's Conv1d 'same' string handles this for stride 1.
                # For stride > 1, 'same' is more complex and might not perfectly preserve length.
                # We'll rely on PyTorch's 'same' string if kernel_size is odd or if it's even for specific cases.
                # If kernel_size is even, (kernel_size-1)//2 results in asymmetric padding.
                # For simplicity, we will use PyTorch's "same" string directly.
                # Let user handle if it leads to unexpected sequence length changes with stride > 1.
                padding_val = 'same'
            elif isinstance(cnn_padding, int):
                padding_val = cnn_padding
            else:
                padding_val = 0 # Default if not 'same' or int

            cnn_layers.append(nn.Conv1d(in_channels=current_in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding_val))
            cnn_layers.append(nn.BatchNorm1d(out_channels))

            if cnn_activation.lower() == "relu":
                cnn_layers.append(nn.ReLU())
            elif cnn_activation.lower() == "tanh":
                cnn_layers.append(nn.Tanh())
            elif cnn_activation.lower() == "leaky_relu":
                cnn_layers.append(nn.LeakyReLU())


            current_in_channels = out_channels

        self.cnn_feature_extractor = nn.Sequential(*cnn_layers)
        self.cnn_dropout = nn.Dropout(cnn_dropout_rate)

        # --- BiLSTM Layers ---
        lstm_input_size = current_in_channels
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional_lstm,
            dropout=lstm_dropout_rate if num_lstm_layers > 1 else 0
        )

        lstm_output_features = lstm_hidden_size * 2 if bidirectional_lstm else lstm_hidden_size

        # --- Classifier Head ---
        self.classifier_dropout = nn.Dropout(linear_dropout_rate)
        self.fc = nn.Linear(lstm_output_features, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size/num_features)
        x = x.permute(0, 2, 1)
        # x shape after permute: (batch_size, input_size, seq_len)

        x_cnn = self.cnn_feature_extractor(x)
        # x_cnn shape: (batch_size, cnn_last_out_channels, cnn_output_seq_len)
        
        # If padding='same' in Conv1d leads to an unexpected extra dimension for odd kernels + even input with stride=1
        # This might happen if PyTorch's 'same' for Conv1d pads on both sides, potentially increasing length by 1
        # For example, input_len=100, kernel=5, stride=1, padding='same' -> output_len=100
        # input_len=100, kernel=4, stride=1, padding='same' -> output_len=100 (or 101 depending on PyTorch version/behavior)
        # We need to ensure the sequence length is consistent or handled.
        # For now, assume the output sequence length is as expected or handled by subsequent layers.

        x_cnn_dropout = self.cnn_dropout(x_cnn)
        x_lstm_input = x_cnn_dropout.permute(0, 2, 1)
        # x_lstm_input shape: (batch_size, cnn_output_seq_len, cnn_last_out_channels)

        x_lstm_out, _ = self.lstm(x_lstm_input)
        # x_lstm_out shape: (batch_size, cnn_output_seq_len, lstm_hidden_size * num_directions)

        x_classifier_dropout = self.classifier_dropout(x_lstm_out)
        out_logits = self.fc(x_classifier_dropout)
        # out_logits shape: (batch_size, cnn_output_seq_len, num_classes)

        return out_logits

if __name__ == '__main__':
    # This block needs config.py to run, or hardcoded values for testing
    import config # Temporary import for testing
    print("Testing BiLSTM-CNN 1D model definition...")
    model = BiLSTMCnn1D(
        input_size=config.NUM_FEATURES,
        num_classes=config.NUM_CLASSES,
        cnn_out_channels=config.CNN_OUT_CHANNELS,
        cnn_kernel_sizes=config.CNN_KERNEL_SIZES,
        cnn_strides=config.CNN_STRIDES,
        cnn_padding=config.CNN_PADDING,
        cnn_activation=config.CNN_ACTIVATION,
        cnn_dropout_rate=config.CNN_DROPOUT,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        num_lstm_layers=config.NUM_LSTM_LAYERS,
        lstm_dropout_rate=config.LSTM_DROPOUT,
        bidirectional_lstm=config.BIDIRECTIONAL_LSTM,
        linear_dropout_rate=config.LINEAR_DROPOUT
    ).to(config.DEVICE)
    
    print(model)
    
    batch_size = config.BATCH_SIZE
    seq_len = config.SEQUENCE_LENGTH
    dummy_input = torch.randn(batch_size, seq_len, config.NUM_FEATURES).to(config.DEVICE)
    
    output = model(dummy_input)
    print(f"\nDummy input shape: {dummy_input.shape}")
    print(f"Model output shape: {output.shape}") 

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")