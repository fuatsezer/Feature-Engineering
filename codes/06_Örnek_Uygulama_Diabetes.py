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

df = pd.read_csv("codes/datasets/diabetes.csv")

######################################
# Feature Engineering: Diabets
######################################

######################################
# Görevler
######################################

#####################################
# Görev 1: Keşifci Veri Analizi
#####################################


#####################################
# GENEL RESİM
#####################################

def check_df(dataframe,head=5):
    print("##################### Shape ##########################")
    print(dataframe.shape)
    print("########################## Types ###########################")
    print(dataframe.dtypes)
    print("########################## Head ###########################")
    print(dataframe.head(head))
    print("########################## Tail ###########################")
    print(dataframe.tail(head))
    print("########################## NA ###########################")
    print(dataframe.isnull().sum())
    print("########################## Quantiles ###########################")
    print(dataframe.quantile([0,0.05,0.5,0.95,0.99,1]).T)

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
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#################################")
    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show()

cat_summary(df,"Outcome")

########################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
########################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df,col, plot=True)


#######################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
#######################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df,"Outcome",col)


########################
# KORELASYON
########################

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix",fontsize=20)
plt.show()

############################
# BASE MODEL KURULUMU
############################

y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 2)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")

# Accuracy: 0.77
# Recall: 0.71
# Precision: 0.59
# F1: 0.64
# Auc: 0.75


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value":model.feature_importances_, "Feature":features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value",y="Feature", data= feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")
plot_importance(rf_model, X)
#############################
# GÖREV 2: FEATURE ENGINEERING
#############################

############################
# EKSİK DEĞER ANALİZİ
############################

# Bir insanda Pregnancies ve Outcome dışındaki değişken değerleri 0 olamayacağı bilinmektedir.
# Bundan dolayı bu değerlerle ilgili aksiyon kararı alınmalıdır. 0 olan değerlere NaN atanabilir.
zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies","Outcome"])]

zero_columns

# sıfır içeren gözlem değerlerini bir ile değiştirme
for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])


# Eksik Gözlem Analizi
df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum()> 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss","ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


na_columns = missing_values_table(df, na_name=True)

# Eksik Değerlerin Bağımlı Değişken ile ilişkisinin incelenmesi
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}),end="\n\n\n")

missing_vs_target(df, "Outcome", na_columns)


# Eksik Değerlerin Doldurulması
for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()


df.isnull().sum()


#############################
# AYKIRI DEĞER ANALİZİ
#############################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] <low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in df.columns:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in df.columns:
    print(col, check_outlier(df, col))


############################
# ÖZELLİK ÇIKARIMI
############################

# Yaş değişkenini kategorilere ayırıp yeni yaş değişkeni oluşturulması
df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 50), "NEW_AGE_CAT"] = "senior"

# BMI 18.5 aşağısı underweight, 18.5 ile 24.9 arası normal, 24.9 ile 29.9 arası Overweight ve 30 üstü obez
df["NEW_BMI"] = pd.cut(x=df["BMI"], bins=[0, 18.5,24.9, 29.9, 100],
                       labels=["Underweight","Healthy","Overweight","Obese"])

# Glükoz değerini kategorik değişkene çevirme
df["NEW_GLUCOSE"] = pd.cut(x=df["Glucose"], bins=[0, 140,200,300], labels=["Normal", "Prediabetes","Diabetes"])


# İnsulin Değeri ile Kategorik değişken türetme
def set_insulin(dataframe, col_name="Insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"

df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)

df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]
df["NEW-GLUCOSE*PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]

# Kolonların büyütülmesi

df.columns = [col.upper() for col in df.columns]

df.head()
df.shape
########################
# ENCODING
########################

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols


for col in binary_cols:
    df = label_encoder(df, col)

df.head()

# One-hot encoding İşlemi
# cat_cols listesini güncelleme işlemi
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]
cat_cols


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols,drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)


df.shape

df.head()

#########################
# Standartlaştırma
########################

num_cols


scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()
df.shape



##############################
# MODELLEME
##############################

y = df["OUTCOME"]
X = df.drop("OUTCOME", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 2)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")

# Accuracy: 0.78
# Recall: 0.71
# Precision: 0.64
# F1: 0.68
# Auc: 0.76

# BASE MODEL
# Accuracy: 0.77
# Recall: 0.71
# Precision: 0.59
# F1: 0.64
# Auc: 0.75

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value":model.feature_importances_, "Feature":features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value",y="Feature", data= feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")
plot_importance(rf_model, X)








