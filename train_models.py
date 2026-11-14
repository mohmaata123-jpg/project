import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# مكتبات تقسيم البيانات ومقاييس التقييم
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# الخوارزميات الثلاث التي اخترتها
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ==========================================
# 1. تحميل وتجهيز البيانات
# ==========================================
df = pd.read_csv('industrial_defects_features.csv')

# فصل الميزات (X) عن فئات العيوب (y)
X = df.iloc[:, :-1].values  # كل الأعمدة ما عدا الأخير
y = df.iloc[:, -1].values   # العمود الأخير (Class)

# تحويل النصوص (أسماء العيوب) إلى أرقام يفهمها XGBoost
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# تقسيم البيانات: 80% للتدريب و 20% للاختبار
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 2. توحيد مقياس الميزات (Feature Scaling) لـ SVM
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. بناء وتدريب النماذج
# ==========================================
print("--- جاري تدريب نموذج SVM ---")
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
print("دقة SVM:", accuracy_score(y_test, svm_pred))

print("\n--- جاري تدريب نموذج XGBoost ---")
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)  # لا يشترط Scaling لـ XGBoost
xgb_pred = xgb_model.predict(X_test)
print("دقة XGBoost:", accuracy_score(y_test, xgb_pred))

print("\n--- جاري تدريب نموذج Decision Tree ---")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
print("دقة Decision Tree:", accuracy_score(y_test, dt_pred))
# ==========================================
# 4. تقييم مفصل (Classification Report & Confusion Matrix)
# ==========================================
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

models_dict = {
    "SVM": svm_model.predict(X_test_scaled),
    "XGBoost": xgb_pred,
    "Decision Tree": dt_pred
}

for name, preds in models_dict.items():
    print(f"\n================ {name} Report ================")
    print(classification_report(y_test, preds))
    
    # رسم مصفوفة الارتباك (Confusion Matrix)
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
