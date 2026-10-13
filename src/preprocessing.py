"""Preprocessing builders. Fit ONLY on training data (pipelines enforce this)."""
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .config import NUM_FEATURES, CAT_FEATURES

def build_preprocessor(use_poly: bool = False) -> ColumnTransformer:
    if use_poly:
        # Polynomial expansion on NUMERIC features only (keeps size manageable:
        # 4 numeric -> 14 poly features), categoricals one-hot as usual.
        num_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ])
    else:
        num_pipe = StandardScaler()
    return ColumnTransformer([
        ("num", num_pipe, NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])

def build_pipeline(regressor, use_poly: bool = False) -> Pipeline:
    return Pipeline([("pre", build_preprocessor(use_poly)), ("reg", regressor)])
