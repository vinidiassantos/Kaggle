import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_preprocessing import load_data, clean_data
from feature_engineering import add_features

def train_and_evaluate(train_path):
    # Carregar dados
    df = load_data(train_path)
    df = clean_data(df)
    df = add_features(df)

    # Selecionar features
    features = ["Pclass", "Sex", "Age", "Fare", "FamilySize", "IsAlone"]
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    X = df[features]
    y = df["Survived"]

    # Treinar modelo
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Avaliar
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Accuracy: {acc:.4f}")

    return model

if __name__ == "__main__":
    train_and_evaluate("data/train.csv")
