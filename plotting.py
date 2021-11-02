from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

def plot_acc(history, losstype, ax = None, xlabel = 'Epoch #'):
    history = history.history
    val_error_key = "val_"+losstype
    history.update({'epoch':list(range(len(history[val_error_key])))})
    history = pd.DataFrame.from_dict(history)

    best_epoch = history.sort_values(by = val_error_key,
                                     ascending = True).iloc[0]['epoch']

    if not ax:
      f, ax = plt.subplots(1,1)
    sns.lineplot(x = 'epoch', y = val_error_key,
                 data = history, label = 'Validation', ax = ax)
    sns.lineplot(x = 'epoch', y = losstype,
                 data = history, label = 'Training', ax = ax)
    ax.axvline(x = best_epoch, linestyle = '--',
               color = 'green', label = 'Best Epoch')  
    ax.legend(loc = 1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(losstype+' (Fraction)')
    
    plt.show()