import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torchvision.models import resnet34 as torchvision_resnet34

class Resnet34LSTM(nn.Module):
    def __init__(self, lstm_hidden_size=256, lstm_num_layers=2, num_classes=2):
        super(Resnet34LSTM, self).__init__()
        
        self.resnet = torchvision_resnet34(pretrained=True)
        self.resnet.fc = nn.Identity()
        
        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        
    def forward(self, x, lengths):
        batch_size, seq_length, c, h, w = x.size()
        
        cnn_features = []
        for t in range(seq_length):
            cnn_out = self.resnet(x[:, t, :, :, :])
            cnn_features.append(cnn_out)
        
        cnn_features = torch.stack(cnn_features, dim=1)
        
        packed_input = rnn_utils.pack_padded_sequence(cnn_features, lengths, batch_first=True, enforce_sorted=False)
        
        lstm_out, (hn, cn) = self.lstm(packed_input)
        
        lstm_out, _ = rnn_utils.pad_packed_sequence(lstm_out, batch_first=True)
        
        out = torch.stack([lstm_out[i, length - 1, :] for i, length in enumerate(lengths)])
        
        out = self.fc(out)
        return out
