import torch

class Operator:

    def __init__(self, A):
        self.A = A

    def __call__(self, x):
        return x @ self.A