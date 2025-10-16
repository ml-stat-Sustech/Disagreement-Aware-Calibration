"""Optimization utilities for calibration."""

import torch
from torch.utils.data import DataLoader, TensorDataset


def loss_function(
    logits: torch.Tensor,
    logits_pre: torch.Tensor,
    temperature: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute the loss that encourages temperature scaling alignment."""
    device = logits.device
    logits_pre = logits_pre.to(device)
    temperature = temperature.to(device)

    probs_tuned = torch.softmax(logits / temperature, dim=-1)
    probs_pre = torch.softmax(logits_pre, dim=-1)

    pred = torch.argmax(torch.softmax(logits, dim=-1), dim=1)
    mask = pred == torch.argmax(probs_pre, dim=1)
    mask_tensor = mask.unsqueeze(1)

    probs_tuned = torch.clamp(probs_tuned, epsilon, 1.0 - epsilon)
    probs_pre = torch.clamp(probs_pre, epsilon, 1.0 - epsilon)

    log_term_masked = -probs_pre * torch.log(probs_tuned)
    log_term = torch.where(mask_tensor, log_term_masked, torch.zeros_like(log_term_masked))

    loss_per_sample = torch.sum(log_term, dim=1)
    return torch.mean(loss_per_sample)


def optimize_temperature(
    cal_logits_pre: torch.Tensor,
    cal_logits_post: torch.Tensor,
    epochs: int = 400,
    batch_size: int = 256,
    learning_rate: float = 0.1,
) -> float:
    """Optimize the temperature parameter using gradient descent."""
    dataset = TensorDataset(cal_logits_pre, cal_logits_post)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    temperature = torch.tensor(1.0, requires_grad=True, dtype=torch.float32)
    optimizer = torch.optim.Adam([temperature], lr=learning_rate)

    for _ in range(epochs):
        for batch_logits_pre, batch_logits_post in dataloader:
            optimizer.zero_grad()
            loss = loss_function(batch_logits_post, batch_logits_pre, temperature)
            loss.backward()
            optimizer.step()

    return temperature.item()
