# step2_train_model.py - UPGRADED
# Uses Gradient Boosting which is far better than Random Forest
# for detecting subtle AI voice artifacts

import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("=" * 60)
print("  SentinelVoice - Model Trainer v3")
print("=" * 60)

if not os.path.exists("data/X_features.npy"):
    print("\nERROR: Run step1_generate_data.py first!")
    exit()


print("\n[1/5] Loading dataset...")
X = np.load("data/X_features.npy")
y = np.load("data/y_labels.npy")
print(f"  Samples: {len(X)} | Features: {X.shape[1]}")
print(f"  Real: {int((y==0).sum())} | Fake: {int((y==1).sum())}")

print("[2/5] Splitting 80/20...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("[3/5] Normalizing features...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print("[4/5] Training Gradient Boosting model...")
print("      (This is much more accurate than Random Forest)")
print("      Takes ~60 seconds on CPU...")

model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.08,
    max_depth=5,
    min_samples_split=4,
    subsample=0.85,
    random_state=42,
    verbose=1
)
model.fit(X_train_s, y_train)

print("\n[5/5] Evaluating...")
y_pred = model.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)

print(f"\n  Accuracy: {acc*100:.1f}%")
print("\n  Confusion Matrix (rows=actual, cols=predicted):")
print("              REAL  FAKE")
cm = confusion_matrix(y_test, y_pred)
print(f"  Actual REAL:  {cm[0][0]:4d}  {cm[0][1]:4d}")
print(f"  Actual FAKE:  {cm[1][0]:4d}  {cm[1][1]:4d}")
print("\n  Full Report:")
print(classification_report(y_test, y_pred,
      target_names=["REAL", "FAKE"]))

# Feature importance — which features matter most
importances = model.feature_importances_
top5_idx = np.argsort(importances)[::-1][:5]
print("  Top 5 most important features (indices):", top5_idx.tolist())

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/detector.pkl")
joblib.dump(scaler, "model/scaler.pkl")
np.save("model/feature_importance.npy", importances)

print("\nModel saved to model/detector.pkl")
print("Next: python step3_app.py")
