import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        stable_z = z - np.max(z)
        exps_z = np.exp(stable_z)
        softmax = exps_z / np.sum(exps_z)

        return np.round(softmax,4)
