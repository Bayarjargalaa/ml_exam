import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data_preprocessing import load_and_prepare_data, create_preprocessor
from src.modeling import (
    define_models, define_models_with_pca,
    get_best_model, get_param_grid
)
from src.utils import (
    save_model, save_report, save_summary, plot_roc_curve, plot_eda,
    generate_analysis_report, write_pca_status_to_summary
)

RANDOM_STATE = 42


def main():
    # 1. Load and clean data
    os.makedirs("data/processed", exist_ok=True)
    df = load_and_prepare_data("data/raw/heart_disease_uci.csv")
    df.to_csv("data/processed/heart_disease_uci_cleaned.csv", index=False)
    print("📁 Цэвэрлэсэн дата хадгалагдлаа: data/processed/heart_disease_uci_cleaned.csv")

    # 2. EDA
    plot_eda(df)

    # 3. Split
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 4. Preprocessing
    preprocessor, _, _ = create_preprocessor(df)

    # 5. Define pipelines (PCA + Normal)
    models_normal = define_models(RANDOM_STATE)
    pipelines = {
        name: Pipeline([("preprocessor", preprocessor), ("classifier", model)])
        for name, model in models_normal.items()
    }

    models_pca = define_models_with_pca(preprocessor, RANDOM_STATE)
    pipelines.update(models_pca)

    # 6. Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_model_name, best_pipeline = get_best_model(X_train, y_train, pipelines, cv)

    # 7. GridSearchCV
    model_key = best_model_name.replace("_PCA", "").replace("LogReg", "LogisticRegression").replace("RF", "RandomForest")
    param_grid = get_param_grid(model_key)
    grid_search = GridSearchCV(best_pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # 8. Evaluation
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # 9. Save results
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    save_model(best_model, "models/heart_disease_best_model.pkl")
    save_report(report, conf_matrix, "outputs/reports/classification_report.txt")
    save_summary(best_model_name, accuracy, grid_search.best_params_, "outputs/reports/summary.txt")

    if hasattr(best_model["classifier"], "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_prob, "outputs/figures/roc_curve.png")

    generate_analysis_report(df, best_model, X_test, y_test)
    write_pca_status_to_summary(best_model_name, best_model)

    print("🎉 Pipeline бүрэн дууслаа.")
    print("✅ Best model:", best_model_name)
    print(f"🎯 Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
