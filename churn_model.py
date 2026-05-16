import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CHURN_PATH = "data/Churn_Modelling.csv"

df = pd.read_csv(CHURN_PATH)
df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)
df["Gender"] = (df["Gender"] == "Male").astype(int)
df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

# Feature Engineering
df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
df["ZeroBalance"] = (df["Balance"] == 0).astype(int)
df["ProductsPerTenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)
df["ActiveWithBalance"] = df["IsActiveMember"] * (df["Balance"] > 0).astype(int)
df["CreditScorePerAge"] = df["CreditScore"] / df["Age"]
df["AgeGroup"] = pd.cut(
    df["Age"], bins=[0, 35, 55, 100], labels=[0, 1, 2]
).astype(int)

# 2. Split Features and Target
X = df.drop(columns=["Exited"])
y = df["Exited"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Fit the Scaler on Training Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Scale test data too for evaluation

# 4. Train the Model on SCALED Data
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train_scaled, y_train)

# 5. Save the artifacts
with open("machine_learning_model.pkl", "wb") as model_file:
    pickle.dump(clf, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler successfully trained and exported!")