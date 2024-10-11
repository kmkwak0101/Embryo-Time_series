import glob
from tqdm import tqdm
import cv2
import pandas as pd
import os 

train_csv_path = '/home/super/301.Personal_Folder/09.kmkwak/embryo/data/lstm_gwak_prepared_train_final.csv'
standard = 286717

df = pd.read_csv(train_csv_path)
label_1 = df[df['label']==1]
label_0 = df[df['label']==0]
total_num_0 = 0
total_num_1 = 0

ver_df = pd.DataFrame()
hor_df = pd.DataFrame()
hor_ver_df = pd.DataFrame()
for i, path in enumerate(label_0['image_path']) :
    image_paths = glob.glob(path + '/*.jpg')
    if total_num_0 * 3 > standard :
        break
    total_num_0 += len(image_paths)
    root_path = '/'.join(path.split('/')[:-1])
    folder_1 = root_path + f'/augmentation/hor_{str(i)}'
    folder_2 = root_path + f'/augmentation/ver_{str(i)}'
    folder_3 = root_path + f'/augmentation/hor_ver_{str(i)}'
    os.makedirs(folder_1, exist_ok=True)
    os.makedirs(folder_2, exist_ok=True)
    os.makedirs(folder_3, exist_ok=True)
    
    for image_path in image_paths :
        image = cv2.imread(image_path)
        hor_flip_image = cv2.flip(image,1)
        ver_flip_image = cv2.flip(image,0)
        hol_ver_flip_image = cv2.flip(image,-1)
        
        cv2.imwrite(os.path.join(folder_1, os.path.basename(image_path)), hor_flip_image)
        cv2.imwrite(os.path.join(folder_2, os.path.basename(image_path)), ver_flip_image)
        cv2.imwrite(os.path.join(folder_3, os.path.basename(image_path)), hol_ver_flip_image)
    
    print(total_num_0)
    
    df = label_0.iloc[[i]]
    df['image_path'] = folder_2
    ver_df = pd.concat([ver_df, df])
    df['image_path'] = folder_1
    hor_df = pd.concat([hor_df, df])
    df['image_path'] = folder_3
    hor_ver_df = pd.concat([hor_ver_df, df])

ver_df.to_csv('/media/super/001.Projects/01.Embryo_Selection/01.Internal/2024_09_03_normal_abnormal_image_crop_data/augmentation/ver.csv', index=False)
hor_df.to_csv('/media/super/001.Projects/01.Embryo_Selection/01.Internal/2024_09_03_normal_abnormal_image_crop_data/augmentation/hor.csv', index=False)
hor_ver_df.to_csv('/media/super/001.Projects/01.Embryo_Selection/01.Internal/2024_09_03_normal_abnormal_image_crop_data/augmentation/horver.csv', index=False)
        