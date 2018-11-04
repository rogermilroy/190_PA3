import torch


def get_predictions(outputs):
    return torch.round(outputs)


def accuracy(outputs, targets):
    """
    Computes the per class accuracy of two 2d tensors.
    :param outputs: 2D Tensor of outputs
    :param targets: 2D Tensor of targets TODO think about how it will work with indices of classes.
    :return: 1D Tensor of accuracies.
    """

    # transpose to align classes
    tout = torch.t(outputs)
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

    # find where the true positives are
    true = outputs.t() + targets.t()
    # find how many there are.
    true = torch.sum(torch.eq(true, 2), 1).to(torch.float)

    # find where false positives and negatives are
    false = outputs.t() - targets.t()
    # filter out false negatives and count
    false = torch.sum(torch.eq(false, 1), 1).to(torch.float)

    sum = true + false

    prec = torch.div(true, sum)

    return prec


def recall(outputs, targets):
    """
    Compute per class recall given by TP / TP + FN
    :param outputs:
    :param targets:
    :return:
    """

    # find where the true positives are
    true = outputs.t() + targets.t()
    # find how many there are.
    true = torch.sum(torch.eq(true, 2), 1).to(torch.float)

    # find where false positives and negatives are
    false = outputs.t() - targets.t()
    # filter out false positives and count
    false = torch.sum(torch.eq(false, 1), -1).to(torch.float)

    sum = true + false

    rec = torch.div(true, sum)

    return rec


def bcr(outputs, targets):
    temp = torch.div(precision(outputs, targets) + recall(outputs, targets), 2)
    return temp

