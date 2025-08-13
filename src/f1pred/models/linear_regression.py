from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def build():
    return Pipeline([
        ("scale", StandardScaler(with_mean=True)),
        ("reg", LinearRegression()),
    ])