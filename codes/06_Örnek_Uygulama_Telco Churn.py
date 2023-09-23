import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder,StandardScaler,RobustScaler
from statsmodels.stats.proportion import proportions_ztest
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import matplotlib
matplotlib.use('TkAgg')
pd.set_option("display.max_columns", None)
#pd.set_option("display.max_rows", 10)
pd.set_option("display.float_format", lambda x: '%.3f' % x)
pd.set_option("display.width", 500)


######################################
# Telco Customer Churn Feature Engineering
######################################


# Problem: Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirmek


# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
        # Adım 1: Genel Resmi inceleyiniz.
        # Adım 2: Numerik ve Kategorik değişkenleri
        # Adım 3: Numerik ve Kategorik Değişkenlerin Analizini yapınız.
        # Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef  değişkene göre numerik değişkenlerin
        # Adım 5: Aykırı gözlem analizi yapınız.
        # Adım 6: Eksik gözlem analizi yapınız.
        # Adım 7: Korelasyon Analizi yapınız.

# GÖREV 2: FEATURE ENGINEERING
        # Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız.
        # işlemleri uygulayabilirsiniz.
        # Adım 2: Yeni değişkenler oluşturunuz.
        # Adım 3: Encoding işlemlerini gerçekleştiriniz.
        # Adım 4: Numerik değişkenler için standartlaştırma yapınız.
        # Adım 5: Model oluşturunuz.

# Gerekli kütüphane ve fonksiyonlar

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder,StandardScaler,RobustScaler
from statsmodels.stats.proportion import proportions_ztest
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('TkAgg')
pd.set_option("display.max_columns", None)
#pd.set_option("display.max_rows", 10)
pd.set_option("display.float_format", lambda x: '%.3f' % x)
pd.set_option("display.width", 500)

df = pd.read_csv("codes/datasets/Telco-Customer-Churn.csv")
df.head()
df.shape
df.info()

# TotalCharges sayısal bir değişken olmalı
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

####################################
# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
####################################

###################################
# GENEL RESİM
###################################

def check_df(dataframe, head=5):
    print("##################################### Shape ###########################")
    print(dataframe.shape)
    print("##################################### Types ###########################")
    print(dataframe.dtypes)
    print("##################################### Head ###########################")
    print(dataframe.head(head))
    print("##################################### Tail ###########################")
    print(dataframe.tail(head))
    print("##################################### NA ###########################")
    print(dataframe.isnull().sum())
    print("##################################### Quantiles ###########################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1],numeric_only=True).T)

check_df(df)


#################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİ YAKALAMA
#################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri Setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içersine numerik görünümlü kategorik değişkenler de dahildir.


    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenilen dataframe
    cat_th: int, optional
        numerik fakat kategorik olan değişkenler için eşik değeri
    car_th: int, optional
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kardinal değişken listesi
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_bat_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#################################################")
    if plot:
        sns.countplot(x= dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)


#####################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
#####################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

###########################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
###########################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)

#################################
# KATEGORİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
#################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

#########################
# KORELASYON
#########################

df[num_cols].corr()


# Korelasyon Matrisi
f, ax = plt.subplots(figsize = [18,13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

df.corrwith(df["Churn"],numeric_only=True).sort_values(ascending=False)

################################
# GÖREV 2: FEATURE ENGINEERING
################################

################################
# EKSİK DEĞER ANALİZİ
################################

df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis= 1, keys=["n_miss","ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

df["TotalCharges"].fillna(df["TotalCharges"].median(),inplace=True)

df.isnull().sum()


###################################
# BASE MODEL KURULUMU
###################################

dff = df.copy()

cat_cols = [col for col in cat_cols if col not in ["Churn"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

dff = one_hot_encoder(dff, cat_cols, drop_first=True)

y = dff["Churn"]
X = dff.drop(["Churn", "customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train,y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 4)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 4)}")
print(f"F1: {round(f1_score(y_pred, y_test), 4)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 4)}")

# Accuracy: 0.7847
# Recall: 0.6331
# Precision: 0.493
# F1: 0.5544
# Auc: 0.7292

##################################
# AYKIRI DEĞER ANALİZİ
##################################

def outlier_threshols(dataframe, col_name,q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_threshols(dataframe, col_name)
    return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)

def replace_with_thresholds(dataframe, variable, q1= 0.05, q3=0.95):
    low_limit, up_limit = outlier_threshols(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Aykırı Değer Analizi ve Baskılama İşleme
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


#############################
# Özellik Çıkarımı
############################

# Tenure değişkeninden yıllık kategori değişken oluşturma
df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12),"NEW_TENURE_YEAR"] = "0-1  Year"
df.loc[(df["tenure"] >= 12) & (df["tenure"] <= 24),"NEW_TENURE_YEAR"] = "1-2  Year"
df.loc[(df["tenure"] >= 24) & (df["tenure"] <= 36),"NEW_TENURE_YEAR"] = "2-3  Year"
df.loc[(df["tenure"] >= 36) & (df["tenure"] <= 48),"NEW_TENURE_YEAR"] = "3-4  Year"
df.loc[(df["tenure"] >= 48) & (df["tenure"] <= 60),"NEW_TENURE_YEAR"] = "4-5  Year"
df.loc[(df["tenure"] >= 60) & (df["tenure"] <= 72),"NEW_TENURE_YEAR"] = "5-6  Year"

# Kontratı 1 veya 2 yıllık müşterileri Engaged olarak belirtme
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)

# Herhangi bir destek, yedek veya koruma almayan kişiler
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Aylık sözleşmesi bulunan ve genç olan müşteriler
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)

# Kişinin Toplam aldığı servis sayısı
df["NEW_TotalServices"] = (df[["PhoneService", "InternetService", "OnlineSecurity",
                               "OnlineBackup", "DeviceProtection", "TechSupport",
                               "StreamingTV", "StreamingMovies"]] == "Yes").sum(axis=1)

# Herhangi bir streaming hizmeti alan kişiler
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis= 1)

# Kişi otomatik ödeme yapıyor mu?
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    label_encoder = LabelEncoder()
    dataframe[binary_col] = label_encoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)


# One-Hot Encoding İşlemi
# cat cols listesinin güncelleme işlemi

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]
cat_cols


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

#####################################
# MODELLEME
#####################################

y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train,y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 4)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 4)}")
print(f"F1: {round(f1_score(y_pred, y_test), 4)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 4)}")


# Accuracy: 0.7856
# Recall: 0.6366
# Precision: 0.4913
# F1: 0.5546
# Auc: 0.7309


# Base Model
# Accuracy: 0.7847
# Recall: 0.6331
# Precision: 0.493
# F1: 0.5544
# Auc: 0.7292

def plot_feature_importance(importance, names, model_type):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data = {"feature_names":feature_names, "feature_importance": feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=["feature_importance"],ascending=False, inplace=True )

    plt.figure(figsize=(15, 10))

    sns.barplot(x= fi_df["feature_importance"], y= fi_df["feature_names"])

    plt.title(model_type + " FEATURE IMPORTANCE")

    plt.xlabel("FEATURE IMPORTANCE")
    plt.ylabel("FEATURE NAMES")
    plt.show()


plot_feature_importance(catboost_model.get_feature_importance(), X.columns, "CATBOOST")



