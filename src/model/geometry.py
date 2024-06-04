from typing import Tuple, List, Dict, Any

import torch
from torch import Tensor

def apply_frame(R:Tensor, t:Tensor, v:Tensor) -> Tensor:
    assert R.shape == v.shape + (3,)
    assert t.shape == v.shape
    assert v.shape[-1] == 3
    return (torch.matmul(R, v.unsqueeze(-1))).squeeze(-1) + t
    
def apply_frame_inverse(R:Tensor, t:Tensor, v:Tensor) -> Tensor:
    assert R.shape == v.shape + (3,)
    assert t.shape == v.shape
    assert v.shape[-1] == 3
    R_inversed = torch.inverse(R)
    return (torch.matmul(R_inversed, (v - t).unsqueeze(-1))).squeeze(-1)

def compose_frames(R1:Tensor, t1:Tensor, R2:Tensor, t2:Tensor) -> Tuple[Tensor, Tensor]:
    assert t1.shape[-1] == 3
    assert R1.shape == t1.shape + (3,)
    assert t2.shape == t1.shape
    assert R2.shape == R1.shape
    return torch.matmul(R1, R2), apply_frame(R1, t1, t2)

def to_local(R:Tensor, t:Tensor, v:Tensor) -> Tensor:
    """ 
    R.shape[:-3] == t.shape[:-2] == v.shape[:-2] <- Let this be "prefix shape"
    For each prefix index in range(prefix shape), there are  
        -n(==t.shape[-2]) coordinate frames represented by (R, t)
        -m(==v.shape[-2]) vectors represented by v to find local coordinates of.
    the goal is to find the tensor of shape (prefix_shape) + (n, m, 3)
    That represents,
    for each prefix index:
        the local coordinates of the m v-vectors w.r.t. n coordinate frames
    """
    n = t.shape[-2]
    m = v.shape[-2]
    prefix_shape = t.shape[:-2]
    assert R.shape == prefix_shape + (n, 3, 3)
    assert t.shape == prefix_shape + (n, 3)
    assert v.shape == prefix_shape + (m, 3)
    
    R_reshaped = R.unsqueeze(-3).broadcast_to(prefix_shape + (n, m, 3, 3))
    t_reshaped = t.unsqueeze(-2).broadcast_to(prefix_shape + (n, m, 3))
    v_reshaped = v.unsqueeze(-3).broadcast_to(prefix_shape + (n, m, 3))
    
    local_v = apply_frame_inverse(R_reshaped, t_reshaped, v_reshaped)
    return local_v
    
