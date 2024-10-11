import glob
from tqdm import tqdm
import cv2
import pandas as pd
import os 
import numpy as np
import random

train_csv_path_1 = '/home/super/301.Personal_Folder/09.kmkwak/embryo/data/lstm_gwak_prepared_train_final.csv'
train_csv_path_2 = '/media/super/001.Projects/01.Embryo_Selection/01.Internal/2024_09_03_normal_abnormal_image_crop_data/augmentation/ver.csv'
train_csv_path_3 = '/media/super/001.Projects/01.Embryo_Selection/01.Internal/2024_09_03_normal_abnormal_image_crop_data/augmentation/hor.csv'
train_csv_path_4 = '/media/super/001.Projects/01.Embryo_Selection/01.Internal/2024_09_03_normal_abnormal_image_crop_data/augmentation/horver.csv'

df = pd.concat([
    # pd.read_csv(train_csv_path_1),
    pd.read_csv(train_csv_path_2),
    pd.read_csv(train_csv_path_3),
    pd.read_csv(train_csv_path_4)
])
df.reset_index(inplace=True)

df = pd.read_csv(train_csv_path_1)
df = df[df['label']==1]

def brightness(gray, val):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = int(random.uniform(-val, val))
    if brightness > 0:
        gray = gray + brightness
    else:
        gray = gray - brightness
    gray = np.clip(gray, 10, 255)
    return gray

def contrast(gray, min_val, max_val):
    #gray = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    alpha = int(random.uniform(min_val, max_val)) # Contrast control
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha)
    return adjusted

def adjusted_bc(image, brightness_val, min_val,max_val) :
    img = brightness(image, brightness_val)
    img = contrast(img, min_val, max_val)
    return img
    

total_df = pd.DataFrame()
for i, row in tqdm(df.iterrows()) :
    image_paths = glob.glob(row['image_path'] + '/*.jpg')
    root_path = '/'.join(row['image_path'].split('/')[:-1])
    folder_1 = root_path + f'/augmentation_normal/b_1_{str(i)}'
    folder_2 = root_path + f'/augmentation_normal/b_2_{str(i)}'
    folder_3 = root_path + f'/augmentation_normal/b_3_{str(i)}'
    folder_4 = root_path + f'/augmentation_normal/b_4_{str(i)}'
    folder_5 = root_path + f'/augmentation_normal/b_5_{str(i)}'
    
    os.makedirs(folder_1, exist_ok=True)
    os.makedirs(folder_2, exist_ok=True)
    os.makedirs(folder_3, exist_ok=True)
    os.makedirs(folder_4, exist_ok=True)
    os.makedirs(folder_5, exist_ok=True)
    
    for image_path in image_paths :
        image = cv2.imread(image_path)
        
        img1 = adjusted_bc(image, 30, 1, 1.5)
        img2 = adjusted_bc(image, 10, 1, 1.2)
        img3 = adjusted_bc(image, 20, 1, 1.3)
        img4 = adjusted_bc(image, 20, 1, 1.5)
        img5 = adjusted_bc(image, 10, 1, 1.5)
        
        cv2.imwrite(os.path.join(folder_1, os.path.basename(image_path)), img1)
        cv2.imwrite(os.path.join(folder_2, os.path.basename(image_path)), img2)
        cv2.imwrite(os.path.join(folder_3, os.path.basename(image_path)), img3)
        cv2.imwrite(os.path.join(folder_4, os.path.basename(image_path)), img4)
        cv2.imwrite(os.path.join(folder_5, os.path.basename(image_path)), img5)
    
    row['image_path'] = folder_1
    total_df = pd.concat([total_df, pd.DataFrame([row])])
    row['image_path'] = folder_2
    total_df = pd.concat([total_df, pd.DataFrame([row])])
    row['image_path'] = folder_3
    total_df = pd.concat([total_df, pd.DataFrame([row])])
    row['image_path'] = folder_4
    total_df = pd.concat([total_df, pd.DataFrame([row])])
    row['image_path'] = folder_5
    total_df = pd.concat([total_df, pd.DataFrame([row])])
    
total_df.to_csv('/media/super/001.Projects/01.Embryo_Selection/01.Internal/2024_09_03_normal_abnormal_image_crop_data/augmentation_normal/b_total.csv', index=False)
        
    
# for i, path in enumerate(label_1['image_path']) :
#     num = len(glob.glob(path + '/*.jpg'))
#     total_num_1 += num
    
# print(total_num_0)
# print(total_num_1)