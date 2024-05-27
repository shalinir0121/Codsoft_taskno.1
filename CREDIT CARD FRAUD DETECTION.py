import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('/mnt/data/creditcard.csv')

# Inspect the dataset
print(df.head())
print(df.info())
print(df.describe())

# Checking for missing values
print(df.isnull().sum())

# Normalizing the amount column
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

# Dropping irrelevant columns if any
# For example, assuming 'Time' might not be relevant for some models:
# df = df.drop(['Time'], axis=1)

# Splitting features and target
X = df.drop('Class', axis=1)
y = df['Class']
# Check class distribution
print(y.value_counts())

# Apply SMOTE for oversampling
smote = SMOTE(sampling_strategy='minority')
X_res, y_res = smote.fit_resample(X, y)
print(y_res.value_counts())
# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
# Train Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predicting on test set
y_pred_log_reg = log_reg.predict(X_test)

# Evaluate Logistic Regression
print("Logistic Regression Metrics:")
print(classification_report(y_test, y_pred_log_reg))
print("ROC-AUC Score:", roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1]))
# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predicting on test set
y_pred_rf = rf.predict(X_test)

# Evaluate Random Forest
print("Random Forest Metrics:")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC Score:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))
