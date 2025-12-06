import numpy as np
import pandas as pd
import os
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib

# ================================
# DOWNLOAD DATASET
# ================================
print("📥 Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
csv_path = os.path.join(path, "creditcard.csv")
df = pd.read_csv(csv_path)
print("✅ Dataset downloaded successfully!\n")

print("Dataset shape:", df.shape)
print("\nClass distribution:\n", df['Class'].value_counts())

# ================================
# VISUALIZATION (NON-BLOCKING)
# ================================
plt.figure()
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Normal Transactions")
plt.savefig("class_distribution.png")
plt.close()
print("\n✅ Plot completed, continuing to preprocessing and training...")

# ================================
# PREPROCESSING
# ================================
X = df.drop(columns=['Class'])
y = df['Class']

scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# ================================
# TRAIN TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ================================
# HANDLE IMBALANCE USING SMOTE
# ================================
print("\n🔄 Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", np.bincount(y_train))
print("After SMOTE:", np.bincount(y_train_res))
print("✅ SMOTE completed!")

# ================================
# MODEL TRAINING
# ================================
print("\n🚀 Starting model training...\n")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=150, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss')
}

roc_scores = []
model_names = []

for name, model in models.items():
    print(f"\n🔹 Training {name} model...")
    model.fit(X_train_res, y_train_res)
    print(f"✅ {name} model trained!")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n📊 {name} Evaluation:\n")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)

    print("ROC AUC:", roc)
    print("PR AUC:", pr)

    roc_scores.append(roc)
    model_names.append(name)

# ================================
# SAVE BEST MODEL
# ================================
joblib.dump(models["XGBoost"], "fraud_model_xgboost.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n💾 Model and scaler saved successfully!")

# ================================
# FINAL RESULT GRAPH (OUTCOME OF RESEARCH)
# ================================
plt.figure()
plt.bar(model_names, roc_scores)
plt.title("Final Model Performance Comparison (ROC AUC)")
plt.xlabel("Models")
plt.ylabel("ROC AUC Score")
plt.ylim(0.9, 1.0)
plt.grid(True)
plt.savefig("model_performance.png")
plt.close()

print("\n📊 Final performance graph saved as 'model_performance.png'")
print("✅ MODEL TRAINING COMPLETE & SAVED SUCCESSFULLY")
