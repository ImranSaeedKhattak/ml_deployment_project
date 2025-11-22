# run_this_once_after_training.py
import joblib
import json
from sklearn.inspection import permutation_importance
import numpy as np

model = joblib.load("model.joblib")
X_test = ...  # your X_test
y_test = ...  # your y_test
feature_names = X_test.columns.tolist()  # if pandas

# Get predictions and metrics
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()

if hasattr(model, "predict_proba"):
    from sklearn.metrics import roc_auc_score
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
else:
    auc = None

# Feature importance
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
else:
    # fallback: permutation importance (slower)
    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    importances = perm.importances_mean

# Top 5
indices = np.argsort(importances)[::-1][:5]
top_features = [(feature_names[i], float(importances[i])) for i in indices]

# Save
json.dump(feature_names, open("feature_names.json", "w"))
json.dump({
    "test_accuracy": float(accuracy),
    "test_auc": float(auc) if auc else None,
    "top_features": top_features
}, open("model_performance.json", "w"), indent=2)