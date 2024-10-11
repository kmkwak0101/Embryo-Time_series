from torch.utils.data import DataLoader
from Resnet152LSTM import Resnet152LSTM
import SequenceDataset
import torchvision.transforms as transforms
import torch
import pandas as pd
from tqdm import tqdm
from Collate import collate_fn
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance

torch.cuda.empty_cache()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dec_cls_model = DecisionTreeClassifier(random_state = 0)
rand_cls_model = RandomForestClassifier(max_depth=2, random_state=0)
adb_cls_model = AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=0)
knn_cls_model = KNeighborsClassifier(n_neighbors = 3)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Resnet152LSTM(num_classes=2)
model_save_path = '/home/super/301.Personal_Folder/09.kmkwak/team_repo/embryo_training/time_series/weights/aug5_256_2/2024-09-29 13 41_0.7400.pt'
model=torch.load(model_save_path)

model.to(device)
test_dataset = SequenceDataset.ImageSequenceDataset(csv_file='/home/super/301.Personal_Folder/09.kmkwak/embryo/data/lstm_gwak_prepared_train_final_expanded_no_mapping_age.csv', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)


num_epochs = 100
true_labels = []
probabilities = []
clinicals = []
model.eval()
clinical_df = pd.DataFrame()
with torch.no_grad():
    for images, labels, lengths, clinical in tqdm(test_dataloader, desc='Testing'):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images, lengths)
        probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        clinicals.append(clinical)

true_labels = np.array(true_labels)
probabilities = np.array(probabilities)
clinical_df = pd.DataFrame(clinicals)
expanded_data = {
    'morphokinetics_m': [],
    'morphological_assessment': [],
    'expansion_rate': [],
    'patient_age': [],
    'morphokinetics_bl': []
}

for index, row in clinical_df.iterrows():
    expanded_data['morphokinetics_m'].extend(row['morphokinetics_m'])
    expanded_data['morphological_assessment'].extend(row['morphological_assessment'])
    expanded_data['expansion_rate'].extend(row['expansion_rate'])
    expanded_data['patient_age'].extend(row['patient_age'])
    expanded_data['morphokinetics_bl'].extend(row['morphokinetics_bl'])

expanded_df = pd.DataFrame(expanded_data)
df = pd.DataFrame(probabilities, columns=['probabilities'])
train_df = pd.concat([expanded_df, df],axis=1)
knn_cls_model.fit(train_df, true_labels)
dec_cls_model.fit(train_df, true_labels)
rand_cls_model.fit(train_df, true_labels)
adb_cls_model.fit(train_df, true_labels)

dec_cls_proba = dec_cls_model.predict_proba(train_df)[:, 1]
rand_cls_proba = rand_cls_model.predict_proba(train_df)[:, 1]
adb_cls_proba = adb_cls_model.predict_proba(train_df)[:, 1]
knn_cls_proba = knn_cls_model.predict_proba(train_df)[:, 1]

dec_auc = roc_auc_score(true_labels, dec_cls_proba)
rand_auc = roc_auc_score(true_labels, rand_cls_proba)
adb_auc = roc_auc_score(true_labels, adb_cls_proba)
knn_auc = roc_auc_score(true_labels, knn_cls_proba)

print(f"Decision Tree ROC AUC: {dec_auc:.4f}")
print(f"Random Forest ROC AUC: {rand_auc:.4f}")
print(f"AdaBoost ROC AUC: {adb_auc:.4f}")
print(f"KNN ROC AUC: {knn_auc:.4f}")

# Permutation feature importance 계산
result = permutation_importance(knn_cls_model, train_df, true_labels, n_repeats=10, random_state=42)

# 결과 출력
importance_scores = result.importances_mean
for i, score in enumerate(importance_scores):
    print(f"Feature {i}: Importance {score}")