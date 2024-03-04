import numpy as np
import torch

import spaces

def normal_spherical_sampler(M, gt_lambda, threshold):
    N, D = M.shape

    results = np.empty_like(M)
    
    mask = np.ones(N, dtype=bool)
    
    if not isinstance(gt_lambda, np.ndarray):
        gt_lambda = np.ones(D) * gt_lambda
    else:
        if gt_lambda.ndim == 1:
            gt_lambda = gt_lambda.reshape((1, -1))

    while np.any(mask):
        deltas = np.random.standard_normal((np.sum(mask), D)) / np.sqrt(gt_lambda)

        new_m = M[mask] + deltas
        new_norm = np.linalg.norm(new_m, axis=-1, keepdims=True)
        new_mask = np.abs(1 - new_norm) < threshold
        new_mask = new_mask[:, 0]

        p_results = results[mask]
        p_results[new_mask] = (new_m / new_norm)[new_mask]
        results[mask] = p_results
        p_mask = mask[mask]
        p_mask[new_mask] = False
        mask[mask] = p_mask

    return results

class ExtendedNSphereSpace(spaces.NSphereSpace):
    def rejected_projected_normal(self, mean, gt_lambda, size, threshold, device="cpu"):
        """Sample from a Normal distribution in R^N and then project back on the sphere.
        Args:
            mean: Value(s) to sample around.
            std: Concentration parameter of the distribution (=standard deviation).
            size: Number of samples to draw.
            device: torch device identifier
        """

        assert len(mean) == size
        assert mean.shape[-1] == self.n

        mean = mean.detach().cpu().numpy()

        if not isinstance(gt_lambda, np.ndarray):
            if not torch.is_tensor(gt_lambda):
                gt_lambda = torch.ones(self.n) * gt_lambda
            if torch.is_tensor(gt_lambda):
                gt_lambda = gt_lambda.detach().cpu().numpy()

        assert mean.shape[1] == self.n
        assert np.allclose(
            np.sqrt((mean ** 2).sum(-1)), np.array([self.r])
        )

        result = normal_spherical_sampler(mean, gt_lambda, threshold)
        result = torch.Tensor(result).to(device)

        # project back on sphere
        # result /= torch.sqrt(torch.sum(result ** 2, dim=-1, keepdim=True))

        return result