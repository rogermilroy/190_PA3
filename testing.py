import torch


def get_predictions(outputs):
    return torch.round(outputs)


def accuracy(outputs, targets):
    """
    Computes the per class accuracy of two 2d tensors.
    :param outputs: 2D Tensor of outputs
    :param targets: 2D Tensor of targets
    :return: 1D Tensor of accuracies.
    """

    # transpose to align classes
    tout = torch.t(get_predictions(outputs))
    ttar = torch.t(targets)

    samples = len(tout[0])

    # perform equality check for correct predictions.
    acc = torch.eq(tout, ttar)
    sum = torch.sum(acc.to(torch.float), 1)

    # calculate the percentage accuracy.
    temp = torch.mul(torch.div(sum, samples), 100.)

    return temp


def precision(outputs, targets):
    """
    Compute the per class precision given by TP / TP + FP
    :param outputs:
    :param targets:
    :return:
    """
    eps = 0.0000000000000000001
    # find where the true positives are
    true = get_predictions(outputs).t() + targets.t()
    # find how many there are.
    true = torch.sum(torch.eq(true, 2), 1).to(torch.float)

    # find where false positives and negatives are
    false = outputs.t() - targets.t()
    # filter out false negatives and count
    false = torch.sum(torch.eq(false, 1), 1).to(torch.float)

    sum = true + false + eps

    prec = torch.mul(torch.div(true, sum), 100.)

    return prec


def recall(outputs, targets):
    """
    Compute per class recall given by TP / TP + FN
    :param outputs:
    :param targets:
    :return:
    """
    eps = 0.0000000000000000001
    # find where the true positives are
    true = get_predictions(outputs).t() + targets.t()
    # find how many there are.
    true = torch.sum(torch.eq(true, 2), 1).to(torch.float)

    # find where false positives and negatives are
    false = outputs.t() - targets.t()
    # filter out false positives and count
    false = torch.sum(torch.eq(false, 1), -1).to(torch.float)

    sum = true + false + eps

    rec = torch.div(true, sum)

    return rec


def bcr(outputs, targets):
    temp = torch.div(precision(outputs, targets) + recall(outputs, targets), 2)
    return temp


def test(model, computing_device, loader, criterion):
    acc, pr, re, balance = None, None, None, None
    total_val_loss = 0.0
    for i, (val_images, val_labels) in enumerate(loader):
        val_images, val_labels = val_images.to(computing_device), val_labels.to(computing_device)
        val_out = model(val_images)
        val_loss = criterion(val_out, val_labels)
        total_val_loss += float(val_loss)

        if i == 0:
            acc = torch.zeros_like(val_labels[0], dtype=torch.float)
            pr = torch.zeros_like(val_labels[0], dtype=torch.float)
            re = torch.zeros_like(val_labels[0], dtype=torch.float)
            balance = torch.zeros_like(val_labels[0], dtype=torch.float)
        acc += accuracy(val_out, val_labels)
        pr += precision(val_out, val_labels)
        re += recall(val_out, val_labels)
        balance += bcr(val_out, val_labels)

    avg_val_loss = total_val_loss / float(i)
    acc /= float(i)
    pr /= float(i)
    re /= float(i)
    balance /= float(i)
    return (total_val_loss, avg_val_loss, acc, pr, re, balance)

