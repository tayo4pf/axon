from enum import Enum

class Optimizer(Enum):
    SGD = "SGD"
    SGD_WM = "SGD_WM"
    NAG = "NAG"
    AdaGrad = "AdaGrad"
    AdaDelta = "AdaDelta"
    AdaM = "AdaM"