from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

def plot_acc(history, losstype, ax = None, xlabel = 'Epoch #', save=False):
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
    
    if save:
      plt.savefig("loss_curve.png")
    else:
      plt.show()


def plot_predictions(trained_model, testgen):
  for index in range(8):
    preds = trained_model.predict(testgen[index][0])

    fig, axs = plt.subplots(nrows=3, ncols=8, figsize=(16, 12))
    counter = 0
    for i in range(8):
        ax = axs[0,counter]
        ax.axis('off')
        ax.imshow(testgen[index][0][i,:,:,0], cmap='gray')  # , vmax=0.8
        ax.set_title(f"Center {counter}")
        ax = axs[1,counter]
        ax.axis('off')
        ax.imshow(testgen[index][1][i,:,:,0], vmin=0.01, vmax=0.4)
        ax.set_title(f"Ground Truth Green {counter}")
        ax = axs[2,counter]
        ax.axis('off')
        ax.set_title(f"Prediction Green {counter}")
        ax.imshow(preds[i], vmin=0.01, vmax=0.4)  # , vmin=0.01, vmax=1
        counter += 1
    plt.savefig(str(index)+".png")