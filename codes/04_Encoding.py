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


######################################
# 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
######################################

######################################
# Label Encoding & Binary Encoding
######################################

df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]
le.inverse_transform([0,1])
le.classes_

def label_encoder(dataframe, binary_col):
    label_encoder = LabelEncoder()
    dataframe[binary_col] = label_encoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtype not in [int, float,"int64","int32","float32","float64"]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df,col)



df = load_application_train()

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float,"int64","int32","float32","float64"]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df,col)

df[binary_cols]

##########################
# One-Hot Encoding
##########################

df = load()
df.head()
df["Embarked"].value_counts()

pd.get_dummies(df, columns=["Embarked"],drop_first=True).head()

pd.get_dummies(df, columns=["Embarked"],dummy_na=True).head()

pd.get_dummies(df, columns=["Sex","Embarked"], drop_first=True).head()


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = load()

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

# cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 > df[col].nunique() > 2]


for col in ohe_cols:
    one_hot_encoder(df,ohe_cols).head()

###########################
# Rare Encoding
###########################

# 1. Kategorik değişkenlerin azlık çokluk durumunu analiz etme
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.

################
# # 1. Kategorik değişkenlerin azlık çokluk durumunu analiz etme
################

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()


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
    print({col_name: dataframe[col_name].value_counts(),
           "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)})
    print("#######################################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df,col)


###################
# # 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
##################

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col,":",len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df,"TARGET", cat_cols)


#######################
# 3. Rare encoder yazacağız.
#######################

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df



new_df = rare_encoder(df,0.01)

rare_analyser(new_df, "TARGET", cat_cols)

#################################
# Feature Scaling (Özellik Ölçeklendirme)
#################################

#############################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar. Standard sapmaya böl. z = (x - u) / s
#############################

df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()

#############################
# Robust: Medyanı çıkar. IQR'a böl. z = (x - median) / iqr
#############################

rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T
df.head()

#############################
# MinMaxScaler: Verilen 2 değer arasına değişken dönüşümü
#############################

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])

df.describe().T

#################
# Numeric to Categorical: Sayısal Değişkenleri Kategork Değişkenlere Çevirme
# Binning
#################


df["Age_qcut"] = pd.qcut(df["Age"], 5)


df.head()