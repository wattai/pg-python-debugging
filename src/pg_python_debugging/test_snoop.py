import snoop
import numpy as np
from sklearn.linear_model import LogisticRegression as Model

@snoop
def f(x):
    out = x + 3
    return out

@snoop
def fit(model_trainer, X, y, random_state: int = 0):
    return model_trainer(random_state=random_state).fit(X, y)

@snoop
def predict(trained_model, X):
    return trained_model.predict(X)

if __name__ == "__main__":
    x = 13
    print(f(x))

    X = np.random.rand(100, 2)
    y = np.random.randint(low=0, high=2, size=(100))

    trained_model = fit(Model, X, y)
    y_pred = predict(trained_model, X)

    print(y_pred)

