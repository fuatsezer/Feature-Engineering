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
#pd.set_option("display.max_rows", 10)
pd.set_option("display.float_format", lambda x: '%.3f' % x)
pd.set_option("display.width", 500)

def load_application_train():
    data = pd.read_csv("codes/datasets/application_train.csv")
    return data



def load():
    data = pd.read_csv("codes/datasets/titanic.csv")
    return data

df = load()
df.head()

####################################
# Eksik Değerlerin Yakalanması
####################################

# Eksik Değer var mı yok mu sorgusu
df.isnull().values.any()

# Değişkenlerdeki eksik değer sayısı
df.isnull().sum()

# Değişkenlerdeki tam değer sayısı
df.notnull().sum()

# veri setindeki toplam eksik değer sayısı
df.isnull().sum().sum()

# en az bir tane eksik değere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)].head()

# tam olan gözlem birimleri
df[df.notnull().all(axis=1)].head()

# azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

# eksik değer oranını hesaplamak
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

def missing_values_table(dataframe, na_name=False):
    na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_cols].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss","ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_cols


missing_values_table(df)

######################################
# Eksik Değer Problemini Çözme
######################################

missing_values_table(df)


#########################
# Çözüm 1: Hızlıca Silmek
#########################

df.dropna().shape

###########################
# Çözüm 2: Basit Atama Yöntemleri ile Doldurmak
###########################

df["Age"].fillna(df["Age"].mean()).isnull().sum()
df["Age"].fillna(df["Age"].median()).isnull().sum()
df["Age"].fillna(0).isnull().sum()


df.apply(lambda x: x.fillna(x.mean()) if x.dtype !="O" else x, axis=0)

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype !="O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)

df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()

df["Embarked"].fillna("missing")

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype !="O" else x, axis=0)

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and x.nunique() <= 10) else x, axis=0).isnull().sum()


#################
# Kategorik Değişken Kırılımında Değer Atama
#################

df.groupby("Sex")["Age"].mean()

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()


##################
# Çözüm 3: Tahmine Dayalı Atama İşlemi
##################

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

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

dff

# Değişkenlerin Standartlaştırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()


## knn'in uygulanması
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)

dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()
# opsiyonel
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)




#############################
# Gelişmiş Analizler
##############################

#################################
# Eksik Verilerin Yapısını inceleme
##################################
## Veri setindeki tam olan değerleri getirme
msno.bar(df)
plt.show()

# Verinin değişkenlerdeki eksikliklerine birlikte bakma
msno.matrix(df);
plt.show()

# Nullity correlation
msno.heatmap(df)
plt.show()

####################
# Eksik Değerlerin Bağımlı Değişken ile ilişkisinin İncelenmesi
####################

missing_values_table(df,True)
na_cols = missing_values_table(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN":temp_df.groupby(col)[target].mean(),"Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df,"Survived",na_cols)

