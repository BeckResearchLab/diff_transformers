import torch

def masked_loss(pred, tgt, mask, loss_fn):
    """
    Computes the masked loss, where loss is only computed for non-padded tokens.

    Args:
    - pred: Predictions from the model, shape [batch_size, seq_len, ...].
    - tgt: Target values, shape [batch_size, seq_len, ...].
    - mask: Padding mask, shape [batch_size, seq_len], where 1 indicates valid tokens and 0 indicates padding.
    - loss_fn: The loss function to apply (e.g., nn.MSELoss, nn.CrossEntropyLoss) with reduction='none'.

    Returns:
    - loss: Scalar value of the masked loss.
    """
    raw_loss = loss_fn(pred, tgt)
    mask = mask.unsqueeze(-1).expand_as(raw_loss)
    masked_loss = raw_loss * ~mask
    total_loss = masked_loss.sum() / (~mask).sum()

    return total_loss