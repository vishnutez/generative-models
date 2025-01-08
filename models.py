from torch.nn import Module, Linear, Sequential, ELU
from torch import Tensor, cat, sqrt
from utils import VPNoiseSchedule

# Define the network
class DiffusionNet(Module, VPNoiseSchedule):
    def __init__(self, 
                 beta_min: float = 0.1, 
                 beta_max: float = 10,
                 dim: int = 2, 
                 h: int = 128, 
                 n_layers: int = 4,
                 eps: float = 1e-5,
        ):
        super().__init__()
        VPNoiseSchedule.__init__(self, beta_min=beta_min, beta_max=beta_max)
        self.dim = dim
        self._in = Linear(dim+1, h)
        self._block = Sequential(Linear(h, h), ELU())
        self._out = Linear(h, dim)
        self._backbone = Sequential(*[self._block for _ in range(n_layers)])
        self.net = Sequential(self._in, self._backbone, self._out)
        self.eps = eps
    
    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        return self.net(cat((t, x_t), -1)) # Used to predict noise in step t
    
    def score(self, t: Tensor, x_t: Tensor) -> Tensor:

        # Relationship between noise and score in step t in VP-SDE:  
        # VP-SDE: x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * z_t -> p_{t|0}(x_t | x_0) = N(x_t | sqrt(alpha_t) * x_0, (1-alpha_t)I)
        # score_t = -(x_t - sqrt(alpha_t)*x_0) / (1-alpha_t) = -z_t  / (sqrt(1-alpha_t)) ~ -noise_pred_t / (sqrt(1-alpha_t)) 
        
        return - self(t, x_t) / (sqrt(1-self.alpha(t)) + self.eps)  