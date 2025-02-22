import numpy as np

import sys

sys.path.append('..')

def inc_corr_plot(y, y_pred, wts=None, fake_normalize=True, *, date_col='dtdate'):
    from pd_utils import cum_corr
    corr = cum_corr(y, y_pred, wts, normalized=fake_normalize).reset_index().rename(columns={0: 'corr'})
    corr = corr[[date_col, 'corr']].groupby(date_col).last()['corr']
    if fake_normalize:
        corr = corr * np.arange(len(corr)) / len(corr)

    def cor_plot(ax):
        ax.plot(corr.index, corr)

    from plot_utils import View, Plot
    return Plot(cor_plot)