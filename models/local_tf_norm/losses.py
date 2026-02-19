import torch


def residual_stationarity_loss(
    residual: torch.Tensor, num_chunks: int = 4, eps: float = 1e-6
) -> torch.Tensor:
    # residual: (B, T, C)
    chunks = torch.chunk(residual, chunks=num_chunks, dim=1)
    means = []
    stds = []
    for chunk in chunks:
        means.append(chunk.mean(dim=1))
        stds.append(chunk.std(dim=1, unbiased=False) + eps)
    means = torch.stack(means, dim=1)
    stds = torch.stack(stds, dim=1)
    mean_var = means.var(dim=1, unbiased=False).mean()
    std_var = stds.var(dim=1, unbiased=False).mean()
    return mean_var + std_var
