import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. LOAD DATA
data = pd.read_csv(r"C:\Users\WELCOME\Clustered_Data.csv")

# 2. FEATURES & TARGET
X = data.drop(["Cluster_Label", "Brand"], axis=1)
y = data["Cluster_Label"]

# 3. ENCODE CATEGORICAL FEATURES
le_load = LabelEncoder()
le_machine = LabelEncoder()

X["Load_Type"] = le_load.fit_transform(X["Load_Type"])
X["Machine_Type"] = le_machine.fit_transform(X["Machine_Type"])

# Encode target
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# 4. SAVE FEATURE ORDER
feature_columns = X.columns.tolist()

# 5. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 6. SCALING
scaler = StandardScaler()

X_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=feature_columns,
    index=X_train.index
)

X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_columns,
    index=X_test.index
)

# 7. MODEL TRAINING
model = LogisticRegression(max_iter=500, random_state=42)
model.fit(X_train, y_train)

# 8. EVALUATION
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)
print("\nClassification Report:\n")
print(classification_report(y_test, model.predict(X_test)))

# 9. SAVE PICKLE FILES
joblib.dump(model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_load, "le_load.pkl")
joblib.dump(le_machine, "le_machine.pkl")
joblib.dump(le_target, "target_encoder.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")

print("\n✅ Training completed successfully")





