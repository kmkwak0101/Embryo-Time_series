import torch
import torch.nn.utils.rnn as rnn_utils

def collate_fn(batch):
    images, labels, additional_data = zip(*batch)
    lengths = [img.size(0) for img in images]
    padded_images = rnn_utils.pad_sequence(images, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    additional_data_collated = {key: [d[key] for d in additional_data] for key in additional_data[0].keys()}
    return padded_images, labels, lengths, additional_data_collated