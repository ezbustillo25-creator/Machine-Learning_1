import numpy as np
from abc import ABC, abstractmethod
from scipy.sparse.linalg import svds


class AnalyticSolutionOptimizer:
    # this optimizer doesnt iterate, it just calls the loss function
    # to get the exact weights directly
    def __init__(self):
        self.model = None

    def set_model(self, model):
        self.model = model

    def optimize(self):
        # ask the loss function to compute weights analytically
        self.model.w = self.model.loss_function.analytic_solution(
            self.model.X_train, self.model.y_train
        )


class LossFunctionBase(ABC):
    # every loss function needs to implement these two methods

    @abstractmethod
    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        pass

    @abstractmethod
    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        pass

    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # not all loss functions have a closed form, so raise error by default
        raise NotImplementedError(
            f"{type(self).__name__} does not have an analytic solution"
        )


class MSELoss(LossFunctionBase):
    # mean squared error loss
    # Q(w) = (1/n) * sum((Xw - y)^2)
    # gradient = (2/n) * X^T * (Xw - y)

    def __init__(self, analytic_solution_func=None):
        # optionally pass a custom function for the analytic solution
        self._analytic_solution_func = analytic_solution_func

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        diff = X @ w - y
        return float(np.dot(diff, diff) / len(y))

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        diff = X @ w - y
        return (2.0 / len(y)) * (X.T @ diff)

    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self._analytic_solution_func is not None:
            return self._analytic_solution_func(self, X, y)
        return self._plain_analytic_solution(X, y)

    def _plain_analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # normal equations: w = (X^T X)^-1 X^T y
        return np.linalg.solve(X.T @ X, X.T @ y)

    def _svd_analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # use SVD when X might be rank deficient
        # w = V * Sigma^+ * U^T * y
        n, d = X.shape
        k = min(n, d) - 1
        U, s, Vt = svds(X, k=k, tol=0)
        eps = np.finfo(float).eps * max(n, d) * s.max()
        s_inv = np.where(s > eps, 1.0 / s, 0.0)
        return Vt.T @ (s_inv * (U.T @ y))


class L2Regularization(LossFunctionBase):
    # wraps another loss and adds L2 penalty: lambda/2 * ||w||^2

    def __init__(self, base_loss: LossFunctionBase, mu: float = 1e-3):
        self.base_loss = base_loss
        self.mu = mu

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        return self.base_loss.loss(X, y, w) + 0.5 * self.mu * np.dot(w, w)

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        return self.base_loss.gradient(X, y, w) + self.mu * w

    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # ridge regression closed form
        d = X.shape[1]
        return np.linalg.solve(X.T @ X + self.mu * np.eye(d), X.T @ y)


class LogCoshLoss(LossFunctionBase):
    # log-cosh loss, smoother alternative to MAE
    # L = (1/n) * sum(log(cosh(Xw - y)))

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        diff = X @ w - y
        return float(np.mean(np.log(np.cosh(diff))))

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        diff = X @ w - y
        return (X.T @ np.tanh(diff)) / len(y)


class HuberLoss(LossFunctionBase):
    # huber loss: squared for small errors, linear for big ones
    # controlled by delta threshold

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        r = X @ w - y
        vals = np.where(
            np.abs(r) < self.delta,
            0.5 * r ** 2,
            self.delta * np.abs(r) - 0.5 * self.delta ** 2
        )
        return float(np.mean(vals))

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        r = X @ w - y
        g = np.where(np.abs(r) < self.delta, r, self.delta * np.sign(r))
        return (X.T @ g) / len(y)


class CustomLinearRegression:
    # linear regression model, takes any loss function and optimizer

    def __init__(self, optimizer=None, loss_function: LossFunctionBase = None,
                 tolerance: float = 1e-6, max_iter: int = 1000):
        self.optimizer = optimizer
        self.loss_function = loss_function if loss_function is not None else MSELoss()
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w = None
        self.X_train = None
        self.y_train = None
        self.loss_history = []

        if optimizer is not None and hasattr(optimizer, 'set_model'):
            optimizer.set_model(self)
        if hasattr(optimizer, 'tolerance') and optimizer.tolerance == 1e-6:
            optimizer.tolerance = tolerance
        if hasattr(optimizer, 'max_iter') and optimizer.max_iter == 1000:
            optimizer.max_iter = max_iter

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w

    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.loss_function.gradient(X, y, self.w)

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.loss_function.loss(X, y, self.w)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y
        self.w = np.zeros(X.shape[1])
        self.loss_history = []

        if self.optimizer is not None:
            self.optimizer.optimize()
        else:
            raise ValueError("no optimizer was given")
