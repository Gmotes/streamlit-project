import pickle
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

support = pd.read_csv("PaySim.csv")

# DestType oluşturma
support["DestType"] = support["nameDest"].str[0]

#kolon temizleme
support.drop(["nameOrig", "nameDest"], axis=1, inplace=True)
support.drop(["isFlaggedFraud"], axis=1, inplace=True)

##################################
# FEATURE ENGINEERING - PaySim
##################################
#Balance Difference
support["BalanceDiff"] = (support["oldbalanceOrg"] - support["newbalanceOrig"])

#Destination Balance Difference
support["DestBalanceDiff"] = (support["newbalanceDest"] - support["oldbalanceDest"])

#Balance Error Flag
support["IsBalanceError"] = np.where(
    support["amount"] !=
    (support["oldbalanceOrg"] - support["newbalanceOrig"]),1,0)

#Large Transaction Flag
support["LargeTransaction"] = np.where(
    support["amount"] >
    support["amount"].quantile(0.95),1,0)

#Step Group
support["StepGroup"] = pd.cut(
    support["step"],
    bins=[0, 200, 400, 600, 800],
    labels=["Early", "Mid", "Late", "VeryLate"])

# Log Feature
support["LogAmount"] = np.log1p(support["amount"])

support["LogOldBalanceOrg"] = np.log1p(support["oldbalanceOrg"])

support["LogNewBalanceOrig"] = np.log1p(support["newbalanceOrig"])

support["LogOldBalanceDest"] = np.log1p(support["oldbalanceDest"])

support["LogNewBalanceDest"] = np.log1p(support["newbalanceDest"])

##################################
# ENCODING
##################################
le = LabelEncoder()

support["DestType"] = le.fit_transform(support["DestType"])

#One hot encode
support = pd.get_dummies(support,
                         columns=["type", "StepGroup"],
                         drop_first=True)
##################################
# Scaling
##################################
scaler = RobustScaler()

scale_cols = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "LogAmount",
    "LogOldBalanceOrg",
    "LogNewBalanceOrig",
    "LogOldBalanceDest",
    "LogNewBalanceDest",
    "BalanceDiff",
    "DestBalanceDiff"
]

support[scale_cols] = scaler.fit_transform(support[scale_cols])

##################################
# MODELLEME
##################################
#Train-Test Split
y = support["isFraud"]
X = support.drop(["isFraud"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=17,
    stratify=y)

##################################
# SMOTE
##################################

smote = SMOTE(random_state=17)

X_resampled, y_resampled = smote.fit_resample(
    X_train,
    y_train)

final_model = LGBMClassifier(
    verbose=-1,
    random_state=17,
    n_jobs=-1)


final_model.fit(X_resampled, y_resampled)

##################################
# PICKLE SAVE
##################################

model_package = {
    "model": final_model,
    "features": X.columns.tolist(),
    "scaler": scaler,
    "scale_cols": scale_cols,
    "label_encoder": le
}

with open("fraud_detection_model.pkl", "wb") as file:
    pickle.dump(model_package, file)

print("Fraud model saved successfully!")