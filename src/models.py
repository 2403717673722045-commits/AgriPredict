"""Six candidate regressors + metadata (which need poly preprocessing)."""
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from .config import SEED

def get_models() -> dict:
    # name -> (regressor, use_poly, cv_folds, short explanation)
    return {
        "Linear Regression": (LinearRegression(), False, 5,
            "Straight-line baseline. Fast and interpretable, but underfits non-linear yield relations."),
        "Polynomial Regression": (LinearRegression(), True, 5,
            "Degree-2 curves on numeric features. More flexible than linear, can overfit."),
        "Decision Tree": (DecisionTreeRegressor(random_state=SEED), False, 5,
            "Splits data into rules (e.g. crop = Potatoes). Captures non-linearity, can overfit alone."),
        "Random Forest": (RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1), False, 3,
            "Average of 100 trees. Reduces overfit, handles country/crop effects well."),
        "KNN": (KNeighborsRegressor(n_neighbors=5), False, 5,
            "Predicts from 5 most similar past records. Simple, but slow and sensitive to high-dim one-hot data."),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=SEED), False, 3,
            "Builds trees sequentially fixing errors. Strong but slower to train."),
    }
