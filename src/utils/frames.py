import numpy as np


def norm(v:np.ndarray):
    return np.sqrt(np.dot(v, v))


def to_frame(x1:np.ndarray, x2:np.ndarray, x3:np.ndarray):
    v1 = x3 - x2
    v1 = v1 / norm(v1)
    v2 = x1 - x2
    v2 = v2 - np.dot(v1, v2) * v1
    v2 = v2 / norm(v2)
    v3 = np.cross(v1, v2)
    R = np.stack([v1, v2, v3], axis=1)
    t = x2
    
    return {'R': R, 't': t}


def get_rotation(x1:np.ndarray, x2:np.ndarray, x3:np.ndarray):
    v1 = x3 - x2
    v1 = v1 / norm(v1)
    v2 = x1 - x2
    v2 = v2 - np.dot(v1, v2) * v1
    v2 = v2 / norm(v2)
    v3 = np.cross(v1, v2)
    R = np.stack([v1, v2, v3], axis=1)
    return R

def test():
    x1 = np.array([2., 2., 2.])
    x2 = np.array([1., 1., 1.])
    x3 = np.array([1., 2., 3.])
    print(to_frame(x1, x2, x3))

if __name__ == '__main__':
    test()
    