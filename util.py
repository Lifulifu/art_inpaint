from keras.datasets import mnist, cifar10
from keras.models import load_model
import keras
import numpy as np
import matplotlib.pyplot as plt

def load_mnist():
    (rawtrain, ytrain), (rawtest, ytest) = mnist.load_data()
    Ntrain = rawtrain.shape[0]
    Ntest = rawtest.shape[0]

    # zero-pad 28*28 img into 32*32 for convenience
    # and normalize
    rawtrain = np.pad(rawtrain, ((0,0),(2,2),(2,2)), 'constant') / 256.
    rawtest = np.pad(rawtest, ((0,0),(2,2),(2,2)), 'constant') / 256.

    #x = left half, y = right half
    xtrain = rawtrain[:, :, :16, np.newaxis]
    ytrain = rawtrain[:, :, 16:, np.newaxis]

    xtest = rawtest[:, :, :16, np.newaxis]
    ytest = rawtest[:, :, 16:, np.newaxis]

    return xtrain, xtest, ytrain, ytest

def load_cifar10():
    (xtrainRaw, _), (xtestRaw, _) = cifar10.load_data()
    
    xtrain = xtrainRaw[:, :, :16, :]/256
    ytrain = xtrainRaw[:, :, 16:, :]/256
    xtest = xtestRaw[:, :, :16, :]/256
    ytest = xtestRaw[:, :, 16:, :]/256
    
    return xtrain, xtest, ytrain, ytest

def plotter(x, y=None, model_dirs=[], rgb=False, cmap=None, title=True):
    imgs = dict()
    for model_dir in model_dirs:
        if model_dir == 'gt':
            ypred = y
        else:
            model = load_model(model_dir)
            ypred = model.predict(x)
        img_pred = np.concatenate([x, ypred], axis=2)
        imgs[model_dir] = img_pred
        print(f'"{model_dir}" prediction done.')
    
    fig, axs = plt.subplots(len(x), len(imgs), figsize=(15,15))
    fig.subplots_adjust(wspace=0)

    for i in range(len(x)):
        for j, m in enumerate(model_dirs):
            if title and i == 0:
                axs[i, j].set_title(m)
            if rgb:
                axs[i, j].imshow(imgs[m][i,:,:,:], cmap=cmap)
            else:
                axs[i, j].imshow(imgs[m][i,:,:,0], cmap=cmap)
    fig.show()
    
def random_plotter(x, y=None, model_dirs=[], rgb=False, title=True, dpi=100, im_w=32, im_h=32, shuffle=False):
    nrows, ncols = len(x), len(model_dirs)
    imgs = []
    for model_dir in model_dirs:
        if model_dir == 'gt':
            ypred = y
        elif model_dir == 'left':
            ypred = np.zeros(x.shape)
        else:
            model = load_model(model_dir)
            ypred = model.predict(x)
        batch_pred = np.concatenate([x, ypred], axis=2)
        imgs.append(batch_pred)
        print(f'"{model_dir}" prediction done.')
    imgs = np.swapaxes(np.array(imgs), 0, 1) # (model, sample, im_h, im_w, channel) -> (sample, model, im_h, im_w, channel) 
    
    if shuffle:
        idxs = [] # suffled indexes
        for i in range(nrows): # for each sample
            idx = np.arange(1, ncols)
            np.random.shuffle(idx)
            idx = np.insert(idx, 0, 0) # don't shuffle the first img of each row
            imgs[i] = imgs[i, idx]
            idxs.append(idx)
    
    w, h = im_w*ncols, im_h*nrows
    plt.figure(dpi=dpi, figsize=(w/dpi, h/dpi))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    count = 1
    for i in range(nrows):
        for j in range(ncols):

            plt.subplot(nrows, ncols, count)
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            if rgb:
                plt.imshow(imgs[i, j, :, :, :])
            else:
                plt.imshow(imgs[i, j, :, :, 0], cmap='gray')

            count += 1

    plt.show()
    
    return idxs