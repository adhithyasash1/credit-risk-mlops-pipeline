import pandas as pd
import pytest

from src.features.preprocess import split_xy, build_preprocessor


def _toy_df():
    return pd.DataFrame({
        "amount": [1.0, 2.0, 3.0, 4.0],
        "grade": ["a", "b", "a", "b"],
        "target": [0, 1, 0, 1],
    })


def test_split_xy_removes_target():
    X, y = split_xy(_toy_df(), "target")
    assert "target" not in X.columns
    assert list(y) == [0, 1, 0, 1]
    assert X.shape == (4, 2)


def test_split_xy_requires_target_column():
    with pytest.raises(ValueError):
        split_xy(_toy_df(), "missing")


def test_build_preprocessor_transforms_to_finite_array():
    X, _ = split_xy(_toy_df(), "target")
    Xt = build_preprocessor(X).fit_transform(X)
    assert Xt.shape[0] == 4
    assert Xt.shape[1] >= 2


def test_preprocessor_handles_unseen_category():
    X, _ = split_xy(_toy_df(), "target")
    pre = build_preprocessor(X).fit(X)
    unseen = pd.DataFrame({"amount": [5.0], "grade": ["z"]})
    out = pre.transform(unseen)
    assert out.shape[0] == 1


def test_build_preprocessor_requires_feature_columns():
    with pytest.raises(ValueError):
        build_preprocessor(pd.DataFrame())
