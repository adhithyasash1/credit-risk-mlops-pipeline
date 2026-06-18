"""
explain.py — global SHAP explainability for the @champion model.

Loads the champion model, computes SHAP values on the test set, and logs
the summary plots to MLflow.

Run from project root:  python3 -m src.training.explain
"""
import matplotlib
matplotlib.use("Agg")          # headless: render to file, no display
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import scipy.sparse
import shap
from sklearn.model_selection import train_test_split

from src.features.preprocess import load_config, load_data_from_bq, split_xy


def main():
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
    mlflow.set_experiment(cfg["mlflow_experiment"])

    # Pull the current champion straight from the registry by alias.
    model = mlflow.sklearn.load_model(f"models:/{cfg['model_name']}@champion")

    df = load_data_from_bq(cfg)
    X, y = split_xy(df, cfg["target_column"])
    _, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Split the Pipeline: SHAP TreeExplainer needs the raw model + transformed X.
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    Xt = prep.transform(X_test)
    if scipy.sparse.issparse(Xt):
        Xt = Xt.toarray()
    feat_names = prep.get_feature_names_out()

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(Xt)

    with mlflow.start_run(run_name="shap-explainability"):
        # Beeswarm: direction + magnitude of each feature's effect
        shap.summary_plot(shap_values, Xt, feature_names=feat_names, show=False)
        plt.savefig("shap_summary.png", dpi=120, bbox_inches="tight")
        mlflow.log_artifact("shap_summary.png")
        plt.close()

        # Bar: mean |SHAP| ranking of feature importance
        shap.summary_plot(shap_values, Xt, feature_names=feat_names,
                          plot_type="bar", show=False)
        plt.savefig("shap_importance_bar.png", dpi=120, bbox_inches="tight")
        mlflow.log_artifact("shap_importance_bar.png")
        plt.close()

    print("Logged SHAP plots to MLflow.")


if __name__ == "__main__":
    main()
