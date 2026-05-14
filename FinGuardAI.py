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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
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
# ENCODING
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
#FEATURE ENGINEERING
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

# Scale Etmek
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