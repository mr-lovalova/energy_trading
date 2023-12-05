import torch
import math


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    losses = []
    for batch, (*X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(*X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        losses.append(loss)
    avg_loss = sum(losses) / num_batches
    return avg_loss


def validation_loop(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)

    test_loss_values = []

    with torch.no_grad():
        for *X, y in dataloader:
            pred = model(*X)
            test_loss = loss_fn(pred, y).item()
            test_loss_values.append(test_loss)
    avg_test_loss = sum(test_loss_values) / num_batches
    print(f"Test Error: \n Avg loss: {avg_test_loss:>8f} \n")

    return avg_test_loss


def test_loop(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss_values = []
    avg_test_loss = []

    with torch.no_grad():
        for *X, y in dataloader:
            pred = model(*X)
            test_loss = loss_fn(pred, y)  # .item()
            test_loss = torch.sqrt(test_loss)
            test_loss_values.append(torch.mean(test_loss, dim=0))
            avg_test_loss.append(torch.mean(test_loss))
    # avg_test_loss = sum(test_loss_values) / num_batches
    avg_test_loss = sum(avg_test_loss) / num_batches
    print(f"Test Error: \n Avg loss: {avg_test_loss:>8f} \n")

    return avg_test_loss, test_loss_values
