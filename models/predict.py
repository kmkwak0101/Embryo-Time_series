import torch
from torch.utils.data import DataLoader
from Resnet152LSTM import Resnet152LSTM
import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
import json

def ImageFolderDataset(img_list):
        images = []
        for img in img_list:
            image = np.array(img)
            image = np.repeat(image[..., np.newaxis], 3, -1)
            image = torch.tensor(image, dtype=torch.float32) 
            image = image.permute(0,3,1,2).squeeze(0)
            images.append(image)
        images = torch.stack(images, dim=0)
        images = images.unsqueeze(0)
        return images

def sequence_dataset_classify(data) : 
    """
    Classifies an image sequence as 'Normal' or 'Abnormal' based on a pretrained ResNet152-LSTM model
    and an XGBoost classifier.

    Args:
        data (dict): A dictionary containing the following keys:
            - 'model_save_path' (str): Path to the saved ResNet152-LSTM model weights.
            - 'img_list' (list): A list with a shape of (number of images, 1, 224, 224)
            - 'morphokinetics_m' (int)
            - 'morphological_assessment' (int)
            - 'expansion_rate' (float)
            - 'patient_age' (int)
            - 'morphokinetics_bl' (int)

    Returns:
        dict: A dictionary containing classification results:
            - 'class' (str): Predicted class ('Normal' or 'Abnormal').
            - 'nor_prob' (float): Probability of the 'Normal' class from the XGBoost classifier.
            - 'ab_prob' (float): Probability of the 'Abnormal' class from the XGBoost classifier.
    """
    
    xgb_cls_model = xgb.XGBClassifier()
    ### If you want to increase the score, you need to change the xgb_cls_model's path.
    xgb_cls_model.load_model('./xgb_cls_model_auc_0.85.json')
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Resnet152LSTM(num_classes=2)
    model = torch.load(data['model_save_path'])
    model.to(device)
    model.eval()

    test_dataset = ImageFolderDataset(data['img_list'])
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    probabilities = []
    with torch.no_grad():
        for images in tqdm(test_dataloader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images, lengths=[images.size(1)])
            probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

    expanded_data = {
        'morphokinetics_m': data['morphokinetics_m'],
        'morphological_assessment': data['morphological_assessment'],
        'expansion_rate': data['expansion_rate'],
        'patient_age': data['patient_age'],
        'morphokinetics_bl': data['morphokinetics_bl']
    }
    
    expanded_df = pd.DataFrame([expanded_data])
    df = pd.DataFrame(probabilities, columns=['probabilities'])
    eval_df = pd.concat([expanded_df, df],axis=1)

    proba = xgb_cls_model.predict_proba(eval_df)

    if xgb_cls_model.predict_proba(eval_df)[:,1] >= 0.5 :
        predict = "Abnormal"
    else :
        predict = "Normal"
        
    output = {
        "class" : predict,
        "nor_prob" : proba[0][0],
        "ab_prob" : proba[0][1]
    }
    return output

if __name__ == "__main__" : 
    with open('./normal_image_3.json', "r") as json_file:
        json_data = json.load(json_file)
    
    dummy = {
        'model_save_path' : 'model_path.pt',
        'img_list': json_data['img_list'],
        "morphokinetics_m": 1,
        "morphological_assessment": 2,
        "expansion_rate": 9.38,
        "patient_age": 28,
        "morphokinetics_bl": 3
    }
    
    binary_classify = sequence_dataset_classify(dummy)
    print(binary_classify)