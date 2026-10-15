from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FactorMatrix:
    left: np.ndarray
    right: np.ndarray

    def __post_init__(self):
        left = np.array(self.left, dtype=np.float64, copy=True)
        right = np.array(self.right, dtype=np.float64, copy=True)
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
            raise ValueError("Matrix factors have incompatible dimensions")
        if left.shape[0] < 1 or right.shape[1] < 1:
            raise ValueError("Matrix dimensions are positive")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("Matrix factors contain nonfinite entries")
        left.setflags(write=False)
        right.setflags(write=False)
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)

    @classmethod
    def zeros(cls, rows, columns):
        return cls(np.empty((rows, 0)), np.empty((0, columns)))

    @classmethod
    def from_dense(cls, matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("Expected a finite matrix")
        left, values, right = np.linalg.svd(matrix, full_matrices=False)
        positive = values > 0
        roots = np.sqrt(values[positive])
        return cls(left[:, positive] * roots, roots[:, None] * right[positive])

    @property
    def shape(self):
        return self.left.shape[0], self.right.shape[1]

    @property
    def width(self):
        return self.left.shape[1]

    @property
    def storage(self):
        return self.left.size + self.right.size

    def dense(self):
        return self.left @ self.right

    def scaled(self, coefficient):
        if not np.isfinite(coefficient):
            raise ValueError("Expected a finite coefficient")
        return FactorMatrix(coefficient * self.left, self.right)

    def plus(self, other, coefficient=1.0):
        if self.shape != other.shape or not np.isfinite(coefficient):
            raise ValueError("Matrix addition has incompatible inputs")
        return FactorMatrix(np.concatenate((self.left, coefficient * other.left), axis=1),
                            np.concatenate((self.right, other.right), axis=0))

    def proposal(self, gradient, step):
        if self.shape != gradient.shape or not np.isfinite(step) or step <= 0:
            raise ValueError("Invalid proposal shape or step")
        # Splitting sqrt(step) across both factors adds step times the gradient product.
        root = np.sqrt(step)
        return FactorMatrix(np.concatenate((self.left, root * gradient.left), axis=1),
                            np.concatenate((self.right, root * gradient.right), axis=0))

    def norm_squared(self):
        if self.width == 0:
            return 0.0
        terms = (self.left.T @ self.left) * (self.right @ self.right.T).T
        value = float(np.sum(terms))
        scale = float(np.sum(np.abs(terms)))
        if value <= 64 * np.finfo(float).eps * max(scale, np.finfo(float).tiny):
            _, left_core = np.linalg.qr(self.left, mode="reduced")
            _, right_core = np.linalg.qr(self.right.T, mode="reduced")
            value = float(np.sum((left_core @ right_core.T) ** 2))
        if not np.isfinite(value):
            raise ValueError("Matrix norm is nonfinite")
        return max(0.0, value)

    def inner(self, other):
        if self.shape != other.shape:
            raise ValueError("Inner product has incompatible matrix shapes")
        value = float(np.einsum("ab,ba->", self.left.T @ other.left,
                                other.right @ self.right.T))
        if not np.isfinite(value):
            raise ValueError("Inner product is nonfinite")
        return value

    def spectrum(self):
        if self.width == 0:
            return Spectrum(np.empty((self.shape[0], 0)), np.empty(0),
                            np.empty((0, self.shape[1])))
        # QR reduces the product SVD to a small core without materializing the dense matrix.
        left_basis, left_core = np.linalg.qr(self.left, mode="reduced")
        right_basis, right_core = np.linalg.qr(self.right.T, mode="reduced")
        left, values, right = np.linalg.svd(left_core @ right_core.T, full_matrices=False)
        return Spectrum(left_basis @ left, values, right @ right_basis.T)


@dataclass(frozen=True)
class Spectrum:
    left: np.ndarray
    values: np.ndarray
    right: np.ndarray

    def __post_init__(self):
        left, values, right = (np.array(item, dtype=float, copy=True)
                               for item in (self.left, self.values, self.right))
        if left.ndim != 2 or right.ndim != 2 or values.ndim != 1:
            raise ValueError("Invalid singular decomposition")
        if left.shape[1] != len(values) or right.shape[0] != len(values):
            raise ValueError("Singular decomposition has incompatible dimensions")
        if any(not np.isfinite(item).all() for item in (left, values, right)):
            raise ValueError("Singular decomposition contains nonfinite entries")
        if np.any(values < 0) or np.any(np.diff(values) > 0):
            raise ValueError("Singular values are nonnegative and descending")
        for name, value in [("left", left), ("values", values), ("right", right)]:
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def truncate(self, rank):
        if isinstance(rank, bool) or int(rank) != rank or not 0 <= rank <= len(self.values):
            raise ValueError("Invalid retained rank")
        rank = int(rank)
        roots = np.sqrt(self.values[:rank])
        return FactorMatrix(self.left[:, :rank] * roots,
                            roots[:, None] * self.right[:rank])


def validate_field(field, reference):
    if set(field) != set(reference):
        raise ValueError("The gradient and state block sets differ")
    if any(field[key].shape != reference[key].shape for key in reference):
        raise ValueError("The gradient and state block shapes differ")


def joint_norm(field):
    return float(np.sqrt(sum(matrix.norm_squared() for matrix in field.values())))


def field_difference(first, second):
    validate_field(first, second)
    return {key: first[key].plus(second[key], -1.0) for key in first}
