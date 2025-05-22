import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
pd.set_option('future.no_silent_downcasting', True)
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)

    # 🎯 Target багана үүсгэх
    df["target"] = (df["num"] > 0).astype(int)

    # 🧹 num/id багануудыг хасах
    df.drop(["num", "id"], axis=1, inplace=True, errors="ignore")

    # 🧠 Багана төрлөөр ангилах
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()

    # 🎯 Target-г тооцоолохоос бусад numeric баганууд
    numerical_cols = [col for col in numerical_cols if col != "target"]

    # 🔁 Numerical багануудыг median-р нөхөх
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())

    # 🔁 Categorical багануудыг mode-р нөхөх
    for col in categorical_cols:
        mode = df[col].mode()
        if not mode.empty:
            df[col] = df[col].fillna(mode[0]).infer_objects(copy=False)

    return df



def create_preprocessor(df):
    numerical_features = df.select_dtypes(include=["int64", "float64"]).drop(columns=["target"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object", "bool"]).columns.tolist()

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_transformer, numerical_features),
        ("cat", cat_transformer, categorical_features)
    ])

    return preprocessor, numerical_features, categorical_features
