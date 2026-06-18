"""Global SHAP explainability for the @champion model.

Loads the champion model, computes SHAP values on the test set, and logs
the summary plots to MLflow.

Run from project root:  python3 -m src.training.explain
"""
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")          # headless: render to file, no display
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import scipy.sparse
import shap
from sklearn.model_selection import train_test_split

from src.config import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE
from src.features.feast_loader import load_training_data
from src.features.preprocess import load_config


def main():
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
    mlflow.set_experiment(cfg["mlflow_experiment"])

    # Pull the current champion straight from the registry by alias.
    model = mlflow.sklearn.load_model(f"models:/{cfg['model_name']}@champion")

    X, y = load_training_data(cfg)
    _, X_test, _, _ = train_test_split(
        X, y, test_size=DEFAULT_TEST_SIZE, stratify=y, random_state=DEFAULT_RANDOM_STATE
    )

    # Split the Pipeline: SHAP TreeExplainer needs the raw model + transformed X.
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    Xt = prep.transform(X_test)
    if scipy.sparse.issparse(Xt):
        Xt = Xt.toarray()
    feat_names = prep.get_feature_names_out()

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(Xt)

    with TemporaryDirectory() as tmpdir, mlflow.start_run(run_name="shap-explainability"):
        tmpdir_path = Path(tmpdir)

        # Beeswarm: direction + magnitude of each feature's effect
        summary_path = tmpdir_path / "shap_summary.png"
        shap.summary_plot(shap_values, Xt, feature_names=feat_names, show=False)
        plt.savefig(summary_path, dpi=120, bbox_inches="tight")
        mlflow.log_artifact(str(summary_path))
        plt.close()

        # Bar: mean |SHAP| ranking of feature importance
        bar_path = tmpdir_path / "shap_importance_bar.png"
        shap.summary_plot(shap_values, Xt, feature_names=feat_names,
                          plot_type="bar", show=False)
        plt.savefig(bar_path, dpi=120, bbox_inches="tight")
        mlflow.log_artifact(str(bar_path))
        plt.close()

    print("Logged SHAP plots to MLflow.")


if __name__ == "__main__":
    main()
