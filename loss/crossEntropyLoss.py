import numpy as np

def cross_entropy_loss(y_true, y_pred):
    '''
    Cross-entropy loss function for multi-class classification.
    '''
    y_pred_clipped = np.clip(y_pred, 1e-12, 1. - 1e-12)
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / y_true.shape[0]
    return loss