import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


def score_r2(y_pred, y_true, flatten=True):
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

    if flatten:
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)
    else:
        y_pred = y_pred
        y_true = y_true

    r2_loss = r2_score(y_true, y_pred)

    return r2_loss


def score_pearsonr(y_pred, y_true, flatten=True):
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

    if flatten:
        y_pred = y_pred.reshape(-1).astype(dtype="float")
        y_true = y_true.reshape(-1).astype(dtype="float")
    else:
        y_pred = y_pred.astype(dtype="float")
        y_true = y_true.astype(dtype="float")

    pearsonr_score = np.corrcoef(y_pred, y_true)[0, 1]

    return pearsonr_score


def score_spearmanr(y_pred, y_true, flatten=True):
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

    if flatten:
        y_pred = y_pred.reshape(-1).astype(dtype="float")
        y_true = y_true.reshape(-1).astype(dtype="float")
    else:
        y_pred = y_pred.astype(dtype="float")
        y_true = y_true.astype(dtype="float")

    rho, pval = spearmanr(y_true, y_pred)

    return rho