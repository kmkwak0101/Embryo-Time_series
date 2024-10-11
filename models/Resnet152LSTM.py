import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils import rnn as rnn_utils

class Resnet152LSTM(nn.Module):
    def __init__(self, lstm_hidden_size=256, lstm_num_layers=2, num_classes=2):
        super(Resnet152LSTM, self).__init__()

        self.resnet = models.resnet152(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        self.fc_layers = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(input_size=16, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)
        self.final_fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x, lengths):
        batch_size, seq_length, c, h, w = x.size()

        cnn_features = []
        for t in range(seq_length):
            with torch.no_grad():
                cnn_out = self.resnet(x[:, t, :, :, :]).squeeze(-1).squeeze(-1)
            cnn_out = self.fc_layers(cnn_out)
            cnn_features.append(cnn_out)
        
        cnn_features = torch.stack(cnn_features, dim=1)

        packed_input = rnn_utils.pack_padded_sequence(cnn_features, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, (hn, cn) = self.lstm(packed_input)
        lstm_out, _ = rnn_utils.pad_packed_sequence(lstm_out, batch_first=True)

        out = torch.stack([lstm_out[i, length - 1, :] for i, length in enumerate(lengths)])

        out = self.final_fc(out)
        return out
