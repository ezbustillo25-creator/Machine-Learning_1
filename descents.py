import numpy as np
from abc import ABC, abstractmethod


class LearningRateSchedule(ABC):
    @abstractmethod
    def get_lr(self, iteration: int) -> float:
        pass


class ConstantLR(LearningRateSchedule):
    def __init__(self, lr=0.01):
        self.lr = lr

    def get_lr(self, iteration):
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_=1.0, s0=1.0, p=0.5):
        self.s0 = s0
        self.p = p
        self.lambda_ = lambda_

    def get_lr(self, iteration):
        if iteration == 0:
            return self.s0
        return self.s0 / (1 + self.lambda_ * iteration) ** self.p


class BaseDescent(ABC):
    def __init__(self, lr_schedule=None, tolerance=1e-4, max_iter=1000):
        self.lr_schedule = TimeDecayLR() if lr_schedule is None else lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.iteration = 0
        self.model = None

    def set_model(self, model):
        self.model = model

    @abstractmethod
    def update_weights(self):
        pass

    def step(self):
        self.update_weights()
        self.iteration += 1

    def optimize(self):
        # record initial loss before any update
        self.model.loss_history = []
        self.model.loss_history.append(
            self.model.compute_loss(self.model.X_train, self.model.y_train)
        )

        for _ in range(self.max_iter):
            prev_weights = self.model.w.copy()
            self.step()
            self.model.loss_history.append(
                self.model.compute_loss(self.model.X_train, self.model.y_train)
            )
            # stop early if weights barely changed
            if np.linalg.norm(self.model.w - prev_weights) < self.tolerance:
                break


class VanillaGradientDescent(BaseDescent):
    def __init__(self, lr_schedule=None, tolerance=1e-4, max_iter=1000):
        super().__init__(lr_schedule, tolerance, max_iter)

    def update_weights(self):
        X, y = self.model.X_train, self.model.y_train
        lr = self.lr_schedule.get_lr(self.iteration)
        gradient = self.model.compute_gradients(X, y)
        self.model.w -= lr * gradient


class StochasticGradientDescent(BaseDescent):
    def __init__(self, lr_schedule=None, batch_size=1, tolerance=1e-4, max_iter=1000):
        super().__init__(lr_schedule, tolerance, max_iter)
        self.batch_size = batch_size

    def update_weights(self):
        X, y = self.model.X_train, self.model.y_train
        indices = np.random.choice(len(y), self.batch_size, replace=False)
        X_batch, y_batch = X[indices], y[indices]
        gradient = self.model.compute_gradients(X_batch, y_batch)
        lr = self.lr_schedule.get_lr(self.iteration)
        self.model.w -= lr * gradient


class SAGDescent(BaseDescent):
    def __init__(self, lr_schedule=None, tolerance=1e-4, max_iter=1000):
        super().__init__(lr_schedule, tolerance, max_iter)
        self.grad_memory = None
        self.grad_sum = None

    def update_weights(self):
        X, y = self.model.X_train, self.model.y_train
        num_objects, num_features = X.shape
        if self.grad_memory is None:
            self.grad_memory = np.zeros((num_objects, num_features))
            self.grad_sum = np.zeros(num_features)
        i = np.random.randint(0, num_objects)
        xi, yi = X[i:i+1], y[i:i+1]
        new_grad_i = self.model.compute_gradients(xi, yi)
        self.grad_sum -= self.grad_memory[i]
        self.grad_sum += new_grad_i
        self.grad_memory[i] = new_grad_i
        lr = self.lr_schedule.get_lr(self.iteration)
        gradient = self.grad_sum / num_objects
        self.model.w -= lr * gradient


class MomentumDescent(BaseDescent):
    def __init__(self, lr_schedule=None, beta=0.9, tolerance=1e-4, max_iter=1000):
        super().__init__(lr_schedule, tolerance, max_iter)
        self.beta = beta
        self.velocity = None

    def update_weights(self):
        X, y = self.model.X_train, self.model.y_train
        if self.velocity is None:
            self.velocity = np.zeros_like(self.model.w)
        gradient = self.model.compute_gradients(X, y)
        self.velocity = self.beta * self.velocity + (1 - self.beta) * gradient
        lr = self.lr_schedule.get_lr(self.iteration)
        self.model.w -= lr * self.velocity


class Adam(BaseDescent):
    def __init__(self, lr_schedule=None, beta1=0.9, beta2=0.999, eps=1e-8, tolerance=1e-4, max_iter=1000):
        super().__init__(lr_schedule, tolerance, max_iter)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def update_weights(self):
        X, y = self.model.X_train, self.model.y_train
        if self.m is None:
            self.m = np.zeros_like(self.model.w)
            self.v = np.zeros_like(self.model.w)
        gradient = self.model.compute_gradients(X, y)
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
        t = self.iteration + 1
        m_hat = self.m / (1 - self.beta1**t)
        v_hat = self.v / (1 - self.beta2**t)
        lr = self.lr_schedule.get_lr(self.iteration)
        self.model.w -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AnalyticSolutionOptimizer(BaseDescent):
    def __init__(self, lr_schedule=None, tolerance=1e-4, max_iter=1000):
        super().__init__(lr_schedule, tolerance, max_iter)

    def optimize(self):
        # analytic solution needs only one step, no loop needed
        self.update_weights()
        self.model.loss_history = [
            self.model.compute_loss(self.model.X_train, self.model.y_train)
        ]

    def update_weights(self):
        X, y = self.model.X_train, self.model.y_train
        n_features = X.shape[1]
        reg_matrix = 2 * self.model.l2_coef * np.eye(n_features)
        A = X.T @ X + reg_matrix
        b = X.T @ y
        self.model.w = np.linalg.solve(A, b)
