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
model_save_paths = 'model_path.pt'


test_dataset = SequenceDataset.ImageSequenceDataset(csv_file='test_final.csv', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)




with torch.no_grad():
    for images, labels, lengths, clinical in tqdm(test_dataloader, desc='Testing'):
        images, labels = images.to(device), labels.to(device)
        for model_save_path in model_save_paths :
            true_labels = []
            probabilities = []
            clinicals = []
            clinical_df = pd.DataFrame()
            model=torch.load(model_save_path)
            model.to(device)
            model.eval()
            
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
            folder_name = model_save_path.split('/')[-2]
            
            print(f"{folder_name}------")
            print(f"Decision Tree ROC AUC: {dec_auc:.4f}")
            print(f"Random Forest ROC AUC: {rand_auc:.4f}")
            print(f"AdaBoost ROC AUC: {adb_auc:.4f}")
            print(f"KNN ROC AUC: {knn_auc:.4f}")

# result = permutation_importance(knn_cls_model, train_df, true_labels, n_repeats=10, random_state=42)

# importance_scores = result.importances_mean
# for i, score in enumerate(importance_scores):
#     print(f"Feature {i}: Importance {score}")