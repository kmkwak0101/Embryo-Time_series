from torch.utils.data import DataLoader, ConcatDataset
from Resnet152LSTM import Resnet152LSTM
from Resnet34LSTM import Resnet34LSTM
import SequenceDataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from tqdm import tqdm
from Collate import collate_fn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
import numpy as np
import json
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import roc_curve, auc

torch.cuda.empty_cache()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])
########### 1. load a model ###########
"""
-1 )
If you want to use a pre-trained model,
Just put your model's path in model_save_path and Load a model.
ex.
model_save_path = 'your_model_path.pt'
model = torch.load(model_save_path)

-2 )
If you want to train without pre-trained model,
you just make a model

ex.
model = Resnet152LSTM(num_classes=2)
"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Resnet152LSTM(num_classes=2)
model_save_path = './put_your_path.pt'
model=torch.load(model_save_path)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True)
criterion = nn.CrossEntropyLoss()

########### 2. import the data ###########
"""
You can use a CSV file for training.
Just provide the file path as a list or a string in `csv_file`.
"""
train_dataset = SequenceDataset.ImageSequenceDataset(
    csv_file=[
        './1_put_your_csv_path.csv',
        './2_put_your_csv_path.csv'
              ],
    transform=transform)
# train_dataset = SequenceDataset.ImageSequenceDataset(
#     csv_file='/home/super/301.Personal_Folder/09.kmkwak/embryo/data/lstm_gwak_prepared_train_final.csv',
#     transform=transform)

########### 3. Oversampling (Optional) ###########
"""
If you want to use oversampling,
you need to use `class_weights` and `WeightedRandomSampler`.
"""
# class_weights = torch.tensor([1/132, 1/408], dtype=torch.float).to(device)
# sample_weights = class_weights[train_dataset.dataframe_data['label'].values]
# sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# train_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn, num_workers=4, pin_memory=True,sampler=sampler)
# train_dataset = generate_synthetic_samples(csv_file='/home/super/301.Personal_Folder/09.kmkwak/embryo/data/lstm_gwak_prepared_train_final.csv', transform=transform, k=100, M=0, N=5)


########### 4. Load the Data###########

train_dataloader = DataLoader(train_dataset, batch_size=8,shuffle=True,collate_fn=collate_fn, num_workers=4, pin_memory=True)

test_dataset = SequenceDataset.ImageSequenceDataset(csv_file='./data/test_final.csv', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)


num_epochs = 100
scaler = GradScaler()
train_acc = []
val_acc = []
train_loss = []
val_acc = []
    

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
    
    for images, labels, lengths in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images, lengths)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient for {name}: {param.grad.abs().mean():.4e}")
        #     else:
        #         print(f"No gradient for {name}")
                
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    model.eval()
    probabilities = []
    true_labels = []
    
    validation_loss = 0.0
    with torch.no_grad():
        for images, labels, lengths in tqdm(test_dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, lengths)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    validation_loss /= len(test_dataloader)
    true_labels = np.array(true_labels)
    probabilities = np.array(probabilities)
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    roc_auc = auc(fpr, tpr)

    print(f'Validation Loss: {validation_loss:.4f}')
    scheduler.step(validation_loss)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H %M')
    torch.save(model, f'./weights/aug5_256_2/{now}_{roc_auc:.4f}.pt')
    log = {
        'time' : datetime.datetime.now().strftime('%Y-%m-%d %H %M'),
        'epoch' : epoch + 1,
        'training_loss' : epoch_loss, 
        'validation_loss' : validation_loss,
        'auc' : roc_auc
        }
    with open('./weights/aug5_256_2/aug5_256_2_log.jsonl','a') as f :
        f.write( json.dumps(log, ensure_ascii=False) + "\n" )