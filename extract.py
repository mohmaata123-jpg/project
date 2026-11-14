import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog

# المسار الصحيح لمجلد الصور بناءً على هيكلية مشروعك
train_dir = 'cleaned_dataset/train/images'

# إذا لم يكن موجوداً هنا، يمكنك تجربته في dataset:
if not os.path.exists(train_dir):
    train_dir = 'dataset/train/images'

print(f"جاري قراءة الصور من المسار: {train_dir} ...")

features_list = []
labels_list = []

if os.path.exists(train_dir):
    # إذا كانت الصور مقسمة داخل مجلدات فرعية حسب نوع العيب:
    subfolders = os.listdir(train_dir)
    
    for sub in subfolders:
        sub_path = os.path.join(train_dir, sub)
        
        # إذا كان مجلد فئة أو صور مباشرة
        if os.path.isdir(sub_path):
            class_name = sub
            img_list = os.listdir(sub_path)
        else:
            # إذا كانت الصور مجتمعة مباشرة في images
            class_name = "defect"
            img_list = subfolders
            sub_path = train_dir

        for img_name in img_list:
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(sub_path, img_name) if os.path.isdir(sub_path) else os.path.join(train_dir, img_name)
                
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                    
                img_resized = cv2.resize(img, (64, 64))
                hog_features = hog(img_resized, pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2), visualize=False)
                
                features_list.append(hog_features)
                labels_list.append(class_name)

    if len(features_list) > 0:
        df_features = pd.DataFrame(features_list)
        df_features['label'] = labels_list
        df_features.to_csv('industrial_defects_features.csv', index=False)
        print(f"تم بنجاح! تم استخراج ميزات {len(features_list)} صورة وحفظها في industrial_defects_features.csv")
    else:
        print("تحذير: لم يتم العثور على أي صور صالحة داخل المجلد المحدد.")
else:
    print(f"المجلد {train_dir} غير موجود.")