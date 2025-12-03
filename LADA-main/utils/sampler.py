import torch

def sample_from_prototypes(means, covariances, weights, k=1):
    """
    Sample from a Gaussian Mixture Model (GMM).

    :param means: (torch.Tensor) The means of the GMM, shape = [n_components, n_features]
    :param covariances: (torch.Tensor) The spherical covariances of the GMM, shape = [n_components]
    :param weights: (torch.Tensor) The component weights of the GMM, shape = [n_components]
    :param k: (int) The number of samples generated per component
    :return: (torch.Tensor, torch.Tensor) The generated samples, shape = [n_components * k, n_features], 
             and the sample weights, shape = [n_components * k]
    """
    n_components, n_features = means.shape

    # Step 1: Generate a fixed number of samples per component
    stds = torch.sqrt(covariances).unsqueeze(1).repeat(1, k).reshape(-1, 1)
    repeated_means = means.unsqueeze(1).repeat(1, k, 1).reshape(-1, n_features)
    generated_samples = repeated_means + stds * torch.randn(n_components * k, n_features).to(stds.device)

    # Step 2: Compute sample weights
    weights_per_sample = (weights / k).repeat_interleave(k)

    return generated_samples, weights_per_sample