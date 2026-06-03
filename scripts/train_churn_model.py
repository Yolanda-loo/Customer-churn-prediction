import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv('../data/customer_churn.csv')

# ----------------------------
# Features & Target
# ----------------------------
X = df.drop(['CustomerID', 'Churn'], axis=1)
y = df['Churn']

# Identify columns
num_cols = ['Age', 'Tenure', 'Complaints', 'Revenue']
cat_cols = ['Region']

# ----------------------------
# Preprocessing
# ----------------------------
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(), cat_cols)
])

# ----------------------------
# Model Pipeline
# ----------------------------
model = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', LogisticRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, '../models/churn_model.pkl')

# ----------------------------
# Predictions
# ----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ----------------------------
# Evaluation
# ----------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# ----------------------------
# Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('../visuals/confusion_matrix.png')

# ----------------------------
# ROC Curve
# ----------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig('../visuals/roc_curve.png')

# ----------------------------
# Feature Importance
# ----------------------------
feature_names = (
    num_cols +
    list(model.named_steps['preprocessing']
         .named_transformers_['cat']
         .get_feature_names_out(cat_cols))
)

coefficients = model.named_steps['classifier'].coef_[0]

importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': coefficients
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=importance)
plt.title("Feature Importance")
plt.savefig('../visuals/feature_importance.png')

print("✅ Churn model complete!")
