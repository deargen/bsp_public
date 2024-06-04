from scipy.spatial.transform.rotation import Rotation
import numpy as np 
from math import pi 

def get_random_rotation(n:int, max_angle=None, target_dtype=np.float32):
    """
    return:
        np.array composed of n random rotation matrices
    description:
        If max_angle is None, just an usual random rotations.
        Else, rotation w.r.t. a random axis with uniform angle from [0, max_angle]
    """
    if max_angle is None:
        return Rotation.random(n).as_matrix().astype(target_dtype)
    assert 0 <= max_angle < pi 
    
    theta = np.random.uniform(low=0, high=max_angle, size=n)
    q_ijk_size = np.tan(theta / 2)
    
    q_ijk = np.random.randn(n, 3)
    q_ijk = q_ijk / np.linalg.norm(q_ijk, axis=1)[:, None] * q_ijk_size[:, None]
    
    q = np.concatenate([q_ijk, np.ones((n, 1))], axis=1)
    
    return Rotation.from_quat(q).as_matrix().astype(target_dtype)

if __name__ == '__main__':
    r = get_random_rotation(100000, max_angle=0.1)
    print(Rotation.from_matrix(r).magnitude())