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
from statsmodels.stats.proportion import proportions_ztest
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

###################################
# Feature Extraction
###################################


#################################
# Binary Features: Flag, Bool, True, False
#################################

df["NEW_CABIN_NA_FLAG"] = df["Cabin"].isnull().astype("int")
df.head()

df.groupby("NEW_CABIN_NA_FLAG").agg({"Survived":"mean"})



test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_NA_FLAG"]==1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_NA_FLAG"]==0, "Survived"].sum()],
                                      nobs=[df.loc[df["NEW_CABIN_NA_FLAG"]==1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_NA_FLAG"]==0, "Survived"].shape[0]])


print("Test Stat = %.4f, p-value = %.4f" % (test_stat,pvalue))


df.loc[((df["SibSp"] + df["Parch"]) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df["SibSp"] + df["Parch"]) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived":"mean"})




test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"]=="NO", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"]=="YES", "Survived"].sum()],
                                      nobs=[df.loc[df["NEW_IS_ALONE"]=="NO", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"]=="YES", "Survived"].shape[0]])


print("Test Stat = %.4f, p-value = %.4f" % (test_stat,pvalue))


################################################
# Text'ler Üzerinden Özellik Türetmek
################################################

##########################
# Letter Count
##########################

df["NEW_NAME_COUNT"] = df["Name"].str.len()

df.head()

########################
# Word Count
########################

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

df.head()

####################
# Özel Yapıları Yakalamak
####################

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df.head()

df.groupby("NEW_NAME_DR").agg({"Survived":["mean", "count"]})

###########################
# Regex ile Değişken Türetmek
###########################

df.head()

df["NEW_TITLE"] = df.Name.str.extract(' ([A-Za-z]+)\.',expand=False)

df.head()

df[["NEW_TITLE","Survived","Age"]].groupby(["NEW_TITLE"]).agg({"Survived":"mean","Age":["count","mean"]})


##############################
# Date Değişkenleri Üretmek
##############################

dff =pd.read_csv("codes/datasets/course_reviews.csv")
dff.head()
dff.info()

dff["Timestamp"] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d",exact=False)

dff["year"] = dff["Timestamp"].dt.year
dff["Month"] = dff["Timestamp"].dt.month

# year diff
dff["year_diff"] = date.today().year - dff["Timestamp"].dt.year

dff["day_name"] = dff["Timestamp"].dt.day_name()

dff.head()

###############################
# Feature Interactions (Özellik Etkileşimi)
###############################

df = load()
df.head()

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1







