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


def balance(outputs, targets):
    temp = torch.div(precision(outputs, targets) + recall(outputs, targets), 2)
    return temp


def aggregate_precision(outputs, targets):
    prec = precision(outputs, targets)
    agg = torch.sum(prec) / float(len(prec))
    return agg.item()


def aggregate_recall(outputs, targets):
    rec = recall(outputs, targets)
    agg = torch.sum(rec) / float(len(rec))
    return agg.item()


def aggregate_balance(outputs, targets):
    b = balance(outputs, targets)
    agg = torch.sum(b) / float(len(b))
    return agg.item()


def sub_matrix(output, target):
    """
    Compute confusion matrix for one output, target pair. If target and output are both positive
    then we count it as correct and add one to the column. Else we distribute the error by
    dividing by the number of positive indications.
    :param output:
    :param target:
    :return: 2D Tensor of the confusion matrix.
    """
    dims = len(target)
    pos = torch.sum(output)
    temp = []
    for i in range(dims):
        if target[i] == 1.0 and output[i] == 1.0:
            t = torch.zeros_like(target)
            t[i] = 1.0
            temp.append(t)
        else:
            t = output / pos
            temp.append(t)
    return torch.stack(temp)


def confusion_matrix(outputs, targets, computing_device):
    dims = len(targets[0])
    temp = torch.zeros((dims, dims)).to(computing_device)
    new = []
    for i in range(len(targets)):
        temp += sub_matrix(outputs[i], targets[i])
    for j in range(temp.size()[0]):
        row = temp[j]
        row /= torch.sum(row)
        new.append(row)
    return torch.stack(new)


def test(model, computing_device, loader, criterion):
    acc, pr, re, bal, conf = None, None, None, None, None
    total_val_loss = 0.0
    with torch.no_grad():
        for i, (val_images, val_labels) in enumerate(loader):
            val_images, val_labels = val_images.to(computing_device), val_labels.to(computing_device)
            val_out = model(val_images)
            val_loss = criterion(val_out, val_labels)
            total_val_loss += float(val_loss)

            if i == 0:
                acc = torch.zeros_like(val_labels[0], dtype=torch.float)
                pr = torch.zeros_like(val_labels[0], dtype=torch.float)
                re = torch.zeros_like(val_labels[0], dtype=torch.float)
                bal = torch.zeros_like(val_labels[0], dtype=torch.float)
                conf = torch.zeros((len(val_labels[0]), len(val_labels[0]))).to(computing_device)
            acc += accuracy(val_out, val_labels)
            pr += aggregate_precision(val_out, val_labels)
            re += aggregate_recall(val_out, val_labels)
            bal += aggregate_balance(val_out, val_labels)
            conf += confusion_matrix(val_out, val_labels, computing_device)

        avg_val_loss = total_val_loss / float(i)
        acc /= float(i)
        pr /= float(i)
        re /= float(i)
        bal /= float(i)
        conf /= float(i)
    return (total_val_loss, avg_val_loss, acc, pr, re, bal, conf)


