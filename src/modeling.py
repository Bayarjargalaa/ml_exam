from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA


def define_models(random_state=42):
    """
    PCA-гүй энгийн classifier-уудыг Dictionary хэлбэрээр буцаана.
    """
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "SVC": SVC(probability=True, random_state=random_state),
        "MLPClassifier": MLPClassifier(random_state=random_state, max_iter=1000)
    }
    return models


def define_models_with_pca(preprocessor, random_state=42):
    """
    PCA ашигласан pipeline-уудыг Dictionary хэлбэрээр үүсгэнэ.
    """
    models = {
        "LogReg_PCA": Pipeline([
            ("preprocessor", preprocessor),
            ("pca", PCA(n_components=5)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state))
        ]),
        "RF_PCA": Pipeline([
            ("preprocessor", preprocessor),
            ("pca", PCA(n_components=5)),
            ("classifier", RandomForestClassifier(random_state=random_state))
        ])
    }
    return models


def get_best_model(X_train, y_train, pipelines, cv):
    """
    Cross-validation ашиглан бүх pipeline-уудыг шалгаж,
    хамгийн сайн гүйцэтгэлтэй pipeline-г нэртэй нь хамт буцаана.
    """
    cv_results = {}
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        cv_results[name] = {
            "pipeline": pipe,
            "mean_accuracy": scores.mean(),
            "std_accuracy": scores.std()
        }

    best_name = max(cv_results.items(), key=lambda x: x[1]["mean_accuracy"])[0]
    best_pipeline = cv_results[best_name]["pipeline"]

    print(f"🏆 Best model: {best_name} — Accuracy: {cv_results[best_name]['mean_accuracy']:.4f}")
    return best_name, best_pipeline


def get_param_grid(model_name):
    """
    Загвар тус бүрд тохирох hyperparameter grid-ийг буцаана.
    """
    if model_name == "RandomForest":
        return {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 5, 10]
        }
    elif model_name == "GradientBoosting":
        return {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.01, 0.1],
            "classifier__max_depth": [3, 5]
        }
    elif model_name == "LogisticRegression" or model_name == "LogReg_PCA":
        return {
            "classifier__C": [0.01, 0.1, 1.0, 10.0]
        }
    elif model_name == "DecisionTree":
        return {
            "classifier__max_depth": [None, 5, 10, 15],
            "classifier__min_samples_split": [2, 5, 10]
        }
    elif model_name == "SVC":
        return {
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ['linear', 'rbf']
        }
    elif model_name == "MLPClassifier":
        return {
            "classifier__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "classifier__alpha": [0.0001, 0.001]
        }
    elif model_name == "RF_PCA":
        return {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 5, 10]
        }
    return {}
