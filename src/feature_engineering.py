def add_features(df):
    """Cria novas features úteis para o modelo."""
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
    return df
