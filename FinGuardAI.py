########################################################################################
#                           1.EXPLORATORY DATA ANALYSIS                                #
########################################################################################
# * 1.1.Importing necessary libraries*
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from datetime import date
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

####################################################################################
# * 1.2.Read the dataset*
####################################################################################
primary = pd.read_csv("data/Churn_Modelling.csv")

# * Checking the  primary data*
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Info #####################")
    print(dataframe.head(head))
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1], numeric_only=True).T)

check_df(primary)

support = pd.read_csv("data/PaySim.csv")

# * Checking the  support data*
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Info #####################")
    print(dataframe.head(head))
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1], numeric_only=True).T)

check_df(support)
##################################
# PRIMARY EDA
##################################
##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI - Primary
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols_primary, num_cols_primary, cat_but_car_primary = grab_col_names(primary)

cat_cols_primary
num_cols_primary
cat_but_car_primary

##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ-Primary
##################################

def cat_summary_primary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols_primary:
    cat_summary_primary(primary, col, plot=False)

##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ-Primary
##################################

def num_summary_primary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols_primary:
    num_summary_primary(primary, col, plot=True)


##################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ-Primary
##################################

def target_summary_with_num_primary(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols_primary:
    target_summary_with_num_primary(primary, "Exited", col)

##################################
# KATEGORİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ-Primary
##################################

def target_summary_with_cat_primary(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols_primary:
    target_summary_with_cat_primary(primary, "Exited", col)

##################################
# KORELASYON-Primary
##################################

primary[num_cols_primary].corr()
# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(primary[num_cols_primary].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

primary.select_dtypes(include=[np.number]).corrwith(primary["Exited"]).sort_values(ascending=False)

##################################
# PRIMARY FEATURE ENGINEERING
##################################

##################################
# EKSİK DEĞER ANALİZİ-Primary
##################################
primary.isnull().sum()


##################################
# AYKIRI DEĞER ANALİZİ
##################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Aykırı Değer Analizi ve Baskılama İşlemi

outlier_cols = [col for col in num_cols_primary
                if col not in ["RowNumber", "CustomerId", "Exited", "HasCrCard", "IsActiveMember"]]

for col in outlier_cols:
    print(col, check_outlier(primary, col))

    if check_outlier(primary, col):
        replace_with_thresholds(primary, col)

#kolon temizleme
primary.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

##################################
# ENCODING-Primary
##################################

# Değişkenlerin tiplerine göre ayrılması işlemi
le = LabelEncoder()
primary["Gender"] = le.fit_transform(primary["Gender"])

# One-Hot Encoding İşlemi
# cat_cols listesinin güncelleme işlemi
primary = pd.get_dummies(primary,
                         columns=["Geography"],
                         drop_first=True)

##################################
#FEATURE ENGINEERING-Primary
##################################
# Age Group
primary["AgeGroup"] = pd.cut(primary["Age"],
                             bins=[18, 30, 45, 60, 100],
                             labels=["18-30", "31-45", "46-60", "60+"])

#Balance / Salary Ratio
primary["BalanceSalaryRatio"] = primary["Balance"] / (primary["EstimatedSalary"] + 1)

#Product Activity Score
primary["ProductActivityScore"] = (
        primary["NumOfProducts"] * primary["IsActiveMember"]
)

#Tenure Group
primary["TenureGroup"] = pd.cut(primary["Tenure"],
                                bins=[-1, 2, 5, 10],
                                labels=["0-2", "3-5", "6-10"])
#Customer Health Score
primary["CustomerHealthScore"] = (
        primary["IsActiveMember"] +
        primary["NumOfProducts"] +
        primary["Tenure"] / 10
)
primary.head()
#Credit Score Segment
primary["CreditScoreSegment"] = pd.cut(
    primary["CreditScore"],
    bins=[0, 579, 669, 739, 799, 850],
    labels=["Poor", "Fair", "Good", "VeryGood", "Excellent"]
)
#High Balance Flag
#Yüksek bakiyeli müşterileri ayırır.
primary["HighBalanceFlag"] = np.where(
    primary["Balance"] > primary["Balance"].median(),
    1,
    0
)

#High Salary Flag
primary["HighSalaryFlag"] = np.where(
    primary["EstimatedSalary"] > primary["EstimatedSalary"].median(),
    1,
    0
)
#Active & High Product User
primary["LoyalCustomerFlag"] = np.where(
    (primary["IsActiveMember"] == 1) &
    (primary["NumOfProducts"] >= 2),
    1,
    0
)

#Young Inactive Customer(churn insight)

primary["YoungInactiveFlag"] = np.where(
    (primary["Age"] < 35) &
    (primary["IsActiveMember"] == 0),
    1,
    0
)
primary.head()

cat_cols_primary, num_cols_primary, cat_but_car_primary = grab_col_names(primary)

#Yeni kategorik kolonları encode etmek
ohe_cols = ["AgeGroup",
            "TenureGroup",
            "CreditScoreSegment"]

primary = pd.get_dummies(primary,
                         columns=ohe_cols,
                         drop_first=True)

primary.head()
primary.info()

# Scaling
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

scale_cols = [
    "CreditScore",
    "Age",
    "Balance",
    "EstimatedSalary",
    "BalanceSalaryRatio",
    "CustomerHealthScore",
    "ProductActivityScore"
]

primary[scale_cols] = scaler.fit_transform(primary[scale_cols])

##################################
# MODELLEME-Primary
##################################
primary.head()
#Train-Test Split
y = primary["Exited"]
X = primary.drop(["Exited"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=17,
    stratify=y
)
#stratify=y Train ve test setlerinde: Exited = 0 / 1 oranını korur. train’de %20 churn test’te %20 churn

#MODEL COMPARISON
#class_weight="balanced"
models = [
    ("LR", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ("RF", RandomForestClassifier(random_state=17, class_weight="balanced")),
    ("XGBoost", XGBClassifier()),
    ("LightGBM", LGBMClassifier()),
    ("CatBoost", CatBoostClassifier(verbose=False))
]

from sklearn.model_selection import cross_validate
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

for name, model in models:

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=5,
        scoring=scoring
    )

    print(f"########## {name} ##########")

    print("Accuracy :", round(cv_results["test_accuracy"].mean(), 4))
    print("Precision:", round(cv_results["test_precision"].mean(), 4))
    print("Recall   :", round(cv_results["test_recall"].mean(), 4))
    print("F1 Score :", round(cv_results["test_f1"].mean(), 4))
    print("ROC_AUC  :", round(cv_results["test_roc_auc"].mean(), 4))

    print("\n")

#| Model    | Accuracy   | Recall     | F1         | ROC-AUC    |
#| -------- | ---------- | ---------- | ---------- | ---------- |
#| LR       | 0.7518     | **0.7177** | 0.5408     | 0.8146     |
#| RF       | 0.8586     | 0.4394     | 0.5582     | 0.8478     |
#| XGBoost  | 0.8553     | 0.4963     | 0.5827     | 0.8405     |
#| LightGBM | 0.8595     | 0.4865     | 0.5850     | 0.8587     |
#| CatBoost | **0.8642** | 0.4909     | **0.5955** | **0.8645** |
# En Güçlü Genel Model:CatBoost en yüksek Accuracy en yüksek F1 en yüksek ROC-AUC
# Logistic Regression Recall’da çok güçlü Recall = 0.7177 churn olacak müşterilerin %71’ini yakalıyor.
# CatBoost achieved the best overall classification performance, while Logistic Regression provided the highest recall for churn detection.

#Final Model
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

# SMOTE
smote = SMOTE(random_state=17)

X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# FINAL MODEL
final_model = CatBoostClassifier(
    verbose=False,
    random_state=17
)

# TRAIN
final_model.fit(X_resampled, y_resampled)

# PREDICT
y_pred = final_model.predict(X_test)

y_prob = final_model.predict_proba(X_test)[:, 1]

# METRICS
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC_AUC  :", roc_auc_score(y_test, y_prob))

# Accuracy : 0.855
# Precision: 0.6746031746031746
# Recall   : 0.5564648117839607
# F1 Score : 0.6098654708520179
# ROC_AUC  : 0.8581304519692343

# Threshold Tuning
from sklearn.metrics import f1_score
import numpy as np

thresholds = np.arange(0.1, 0.9, 0.01)

best_threshold = 0
best_f1 = 0

for threshold in thresholds:

    y_pred_threshold = (y_prob >= threshold).astype(int)

    score = f1_score(y_test, y_pred_threshold)

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

print("Best Threshold:", best_threshold)
print("Best F1:", best_f1)

# Best Threshold: 0.3699999999999999
# Best F1: 0.6252945797329144

best_threshold = 0.37

y_pred_final = (y_prob >= best_threshold).astype(int)

print("Accuracy :", accuracy_score(y_test, y_pred_final))
print("Precision:", precision_score(y_test, y_pred_final))
print("Recall   :", recall_score(y_test, y_pred_final))
print("F1 Score :", f1_score(y_test, y_pred_final))
print("ROC_AUC  :", roc_auc_score(y_test, y_prob))

# Accuracy : 0.841
# Precision: 0.6012084592145015
# Recall   : 0.6513911620294599
# F1 Score : 0.6252945797329144
# ROC_AUC  : 0.8581304519692343

# Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": final_model.feature_importances_
}).sort_values("Importance", ascending=False)

feature_importance.head(15)

#                         Feature  Importance
# 2                           Age      23.561
# 5                 NumOfProducts       8.395
# 13          CustomerHealthScore       8.184
# 12         ProductActivityScore       7.218
# 4                       Balance       6.061
# 0                   CreditScore       5.215
# 8               EstimatedSalary       4.585
# 9             Geography_Germany       3.883
# 7                IsActiveMember       3.662
# 11           BalanceSalaryRatio       3.405
# 19               AgeGroup_46-60       3.279
# 23      CreditScoreSegment_Fair       3.038
# 3                        Tenure       2.988
# 24      CreditScoreSegment_Good       2.526
# 25  CreditScoreSegment_VeryGood       2.051


# SHAP

import shap

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

#Hyperparameter tuning

from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier

params = {
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "iterations": [300, 500],
    "l2_leaf_reg": [3, 5, 7]
}

grid = GridSearchCV(
    CatBoostClassifier(verbose=False, random_state=17),
    params,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_resampled, y_resampled)

print(grid.best_params_)
print(grid.best_score_)

# GridSearchCV(cv=3, estimator=CatBoostClassifier(random_state=17, verbose=False),
#              n_jobs=-1,
#              param_grid={'depth': [4, 6, 8], 'iterations': [300, 500],
#                          'l2_leaf_reg': [3, 5, 7],
#                          'learning_rate': [0.03, 0.05, 0.1]},
#              scoring='f1')
# print(grid.best_params_)
# print(grid.best_score_)
# {'depth': 8, 'iterations': 500, 'l2_leaf_reg': 5, 'learning_rate': 0.03}
# 0.8664306322620993

final_model = CatBoostClassifier(
    depth=8,
    iterations=500,
    l2_leaf_reg=5,
    learning_rate=0.03,
    verbose=False,
    random_state=17
)

final_model.fit(X_resampled, y_resampled)

y_prob = final_model.predict_proba(X_test)[:, 1]

best_threshold = 0.37
y_pred_final = (y_prob >= best_threshold).astype(int)

print("Accuracy :", accuracy_score(y_test, y_pred_final))
print("Precision:", precision_score(y_test, y_pred_final))
print("Recall   :", recall_score(y_test, y_pred_final))
print("F1 Score :", f1_score(y_test, y_pred_final))
print("ROC_AUC  :", roc_auc_score(y_test, y_prob))

# Accuracy : 0.8346666666666667
# Precision: 0.5820256776034237
# Recall   : 0.6677577741407529
# F1 Score : 0.6219512195121951
# ROC_AUC  : 0.8567013706438197