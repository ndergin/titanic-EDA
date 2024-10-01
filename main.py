import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

train_df = pd.read_csv("/Users/ndergin/Desktop/titanic-EDA/train.csv")
test_df = pd.read_csv("/Users/ndergin/Desktop/titanic-EDA/test.csv")
test_PassengerId = test_df["PassengerId"]

train_df.columns

train_df.head()

#sayısal sütunlarının özet istatistikleri
train_df.describe()
"""
PassengerId: 891 yolcu var.
Pclass: Ortalama yolcu sınıfı 2.31, 1. sınıf en düşük, 3. sınıf en yüksek sınıf.
Age: Yaş ortalaması 29.7, en küçük yolcu 0.42 yaşında, en yaşlı yolcu 80 yaşında.
SibSp: Kardeş/eş sayısı ortalama 0.52.
Parch: Ebeveyn/çocuk sayısı ortalama 0.38.
Fare: Bilet ücreti ortalama 32.2, en yüksek 512.33.
"""

train_df.info()

"""
RangeIndex: 891 entries, 0 to 890: Veri seti, 0'dan 890'a kadar toplam 891 satırdan oluşur.
Data columns (total 12 columns): Veri setinde 12 sütun bulunur.
Non-Null Count: Her sütunda kaç eksik olmayan değer olduğunu gösterir. Örneğin, Age sütununda 714 eksik olmayan değer bulunurken, Cabin sütununda yalnızca 204 eksik olmayan değer vardır.
Dtype: Her sütunun veri tipi gösterilir. Örneğin, PassengerId bir int64, Age bir float64, Sex ise bir object (yani metin) türündedir.
Memory Usage: DataFrame'in bellekte kapladığı alan yaklaşık 83.7 KB'dir.
"""
"""
float64(2) : Fare, Age
int64(5) : PassengerId, Survived, Pclass, SibSp, Parch,
object(5) : Name, Sex, Ticket, Cabin, Embarked
"""

# UNIVARITE VARIABLE ANALYSIS

# Categorical Variable : Survived, Sex, Pclass, Embarked, Cabin, Name, Ticket, SibSp, Parch
# Numerical Variable : PassengerId, Age, Fare

# Categorical Variable
def bar_plot(variable):
    """
    input: variable ex: Sex
    output: bar plot and value count
    """
    # get feature
    var = train_df[variable]
    
    # count number of categorical variable(value/sample) 
    varValue = var.value_counts()

    # visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequancy")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))

category1 = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]
for c in category1:
    bar_plot(c)

category2 = ["Cabin","Name","Ticket"]
for c in category2:
    print("{}: \n".format(train_df[c].value_counts()))

# Numerical Variable

def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()

numericalVar = ["Fare","PassengerId","Age"]
for n in numericalVar:
    plot_hist(n)

# BASIC DATA ANALYSIS

    # Pclass - Survived
    # Sex - Survived
    # SibSp - Survived
    # Parch - Survived

# Plcass vs Survived 1. sınıf yolcuların hayatta kalma oranı daha yüksek
train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived", ascending = False)
# Sex vs Survived Genellikle kadın yolcuların hayatta kalma oranı daha yüksek
train_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived", ascending = False)
# SibSp vs Survived Yanında 1 kardeşi veya eşi bulunanların hayatta kalma oranı
train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived", ascending = False)
# Parch vs Survived Yannında çocuk ya da ebeveyn olannların hayatta kalma oranı
train_df[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived", ascending = False)
"""
Sonuç olarak:
Daha yüksek sınıftaki yolcuların (1. sınıf) hayatta kalma oranı genellikle daha yüksektir.
Kadın yolcuların hayatta kalma oranı erkeklerden daha yüksektir.
Yanlarında kardeş/eş veya ebeveyn/çocuk bulunan yolcuların hayatta kalma oranı, genellikle yalnız olanlardan daha yüksektir.
"""
# OUTLIER DETECTION
def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        #1st
        Q1 = np.percentile(df[c],25)
        #3rd
        Q3 = np.percentile(df[c],75)
        IQR = Q3 -Q1
        #Outlier step
        outlier_step = IQR * 1.5
        #detect outlier and their indeces
        outlier_list_col = df[(df[c]< Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        #store
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers

train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]

# drop outliers
train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)

# MISSING VALUE

train_df_len = len(train_df)
train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop = True)
train_df.head()

#Find missing value

train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()

#fill missing value

train_df[train_df["Embarked"].isnull()]

train_df.boxplot(column="Fare",by = "Embarked")
plt.show()

train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]

train_df[train_df["Fare"].isnull()]
np.mean(train_df[train_df["Pclass"] == 3]["Fare"])

train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))
train_df[train_df["Fare"].isnull()]
