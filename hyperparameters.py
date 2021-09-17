def adamatch_hyperparams(lr=3e-3, tau=0.9, wd=5e-4, scheduler=True):
    """
    Return a dictionary of hyperparameters for the AdaMatch algorithm.

    Arguments:
    ----------
    lr: float
        Learning rate.

    tau: float
        Weight of the unsupervised loss.

    wd: float
        Weight decay for the optimizer.

    scheduler: bool
        Will use a StepLR learning rate scheduler if set to True.

    Returns:
    --------
    hyperparams: dict
        Dictionary containing the hyperparameters. Can be passed to the `hyperparams` argument on AdaMatch.
    """
    
    hyperparams = {'learning_rate': lr,
                   'tau': tau,
                   'weight_decay': wd,
                   'step_scheduler': scheduler
                   }

    return hyperparams