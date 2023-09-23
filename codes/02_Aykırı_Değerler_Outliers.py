import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder,StandardScaler,RobustScaler
import matplotlib
matplotlib.use('TkAgg')
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: '%.3f' % x)
pd.set_option("display.width", 500)

def load_application_train():
    data = pd.read_csv("codes/datasets/application_train.csv")
    return data

df = load_application_train()
df.head()

def load():
    data = pd.read_csv("codes/datasets/titanic.csv")
    return data

df = load()
df.head()

#################################################
# 1. Outliers (Aykırı Değerler)
#################################################

#########################################
# Aykırı Değerleri Yakalama
#########################################

###################
# Grafik Tekniğiyle Aykırı Değerler
###################

sns.boxplot(df["Age"])
plt.show()


################################################
# Aykırı Değerleri Yakalama
################################################

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)

iqr = q3 - q1

up = q3 + 1.5 *iqr
low = q1 - 1.5 * iqr

df[(df["Age"] < low) | (df["Age"] > up)]

df[(df["Age"] < low) | (df["Age"] > up)].index

#######################
# Aykırı Değer var mı?
#######################

df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)

df[~((df["Age"] < low) | (df["Age"] > up))].any(axis=None)

# 1. Eşik değer belirledik
# 2. Aykırılara eriştik
# 3. Hızlıca aykırı değer var mı yok mu diye sorduk.

####################
# İşlemleri Fonksiyonlaştırma
####################

def outlier_thresholds(dataframe, col_name, q1 = 0.25, q3 = 0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit




def check_outlier(dataframe,col_name):
    low_limit, up_limit = outlier_thresholds(dataframe,col_name)
    return dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None)

check_outlier(df,"Age")

#######################
# grap_col_names
#######################

dff = load_application_train()
dff.head()

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

num_cols = [col for col in num_cols if col not in "PassengerId"]


for col in num_cols:
    print(col, check_outlier(df,col))




cat_cols, num_cols, cat_but_car = grab_col_names(dff)
num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]

for col in num_cols:
    print(col, check_outlier(dff,col))

###################
# Aykırı Değerlerin Kendilerine Erişmek
###################

def grap_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe,col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index



grap_outliers(df,"Age")

grap_outliers(df,"Age",True)



########################
# Aykırı Değer Problemini Çözme
########################


#####################
# Silme
#####################

low, up = outlier_thresholds(df, "Fare")
df.shape

df[~((df["Fare"] < low) | (df["Fare"] > up))].shape

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]


for col in num_cols:
    new_df = remove_outlier(df, col)

df.shape[0] - new_df.shape[0]

####################
# Baskılama Yöntemi (re-assignment with thresholds)
####################

low, up = outlier_thresholds(df, "Fare")

df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"]


df.loc[(df["Fare"] > up), "Fare"] = up

df.loc[(df["Fare"] > low), "Fare"] = low

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df,col))

#######################
# Çok Değikenli Aykırı Değer Analizi: Local Outlier Factor
#######################

df = sns.load_dataset("diamonds")
df = df.select_dtypes(include=["float64","int64"])
df = df.dropna()
df.head()

for col in df.columns:
    print(col, check_outlier(df, col))



low, up = outlier_thresholds(df, "carat")

df[((df["carat"] < low) | (df["carat"] > up))].shape

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_

df_scores[0:5]

np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style=".-")
plt.show()


th = np.sort(df_scores)[3]


df[df_scores < th]

df.describe([0.01, 0.05, 0.75, 0.9, 0.99]).T

df.drop(axis=0, labels=df[df_scores < th].index)


