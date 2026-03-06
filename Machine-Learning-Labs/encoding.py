import pandas as pd 
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("titanic.csv")
df
le = LabelEncoder()
df

df["Gender_encoded"] = le.fit_transform(df["Sex"])
df

cat_col = df.select_dtypes(include=["object"]).columns.tolist()

cat_col

num_col = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

num_col
df.select_dtypes(include=["object"])
cat_col
df1 = pd.read_csv("diamonds.csv")

df1
df1.cut.value_counts()
df1["cut"].value_counts()

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

titanic = pd.read_csv("titanic.csv")

titanic

from sklearn.preprocessing import OneHotEncoder

titanic.isnull().sum()

preprocessor = ColumnTransformer(
    transformers=[
        ("trf1", SimpleImputer(strategy="most_frequent"), ["Sex", "Embarked"]),
        ("cat", OneHotEncoder(), ["Sex", "Embarked"]),
        ("num", SimpleImputer(strategy="mean"), ["Age"]),
    ],
    remainder="passthrough",
)
pipe = Pipeline(steps=[("prepreocess", preprocessor)])

df = pipe.fit_transform(titanic)

df = pd.DataFrame(df)

df.info()

titanic["Age"].max()

titanic["Age"].min()

from sklearn.preprocessing import MinMaxScale

mms = MinMaxScale()

titanic[["Age Scaled"]] = mms.fit_transform(titanic[["Age"]])
titanic["Age Scaled"]
from sklearn.preprocessing import StandardScaler

std = StandardScaler()
df[["age_std"]] = std.fit_transform(titanic[["Age"]])

df["age_std"]