import numpy as np
import natsort
import os
import utils.tools as helper
import utils.load_data as datama
from tqdm import tqdm
import pandas as pd
def summaryCW(modellist =['mlp','lenet','deepxplore'], datalist = ['mnist','fashion_mnist'], adv=['CW']):
    #modellist =['mlp','lenet','deepxplore', 'netinnet','vgg']
    #datalist = ['mnist','fashion_mnist']
    wpath = './cluster_results/model/'
    cifarmodelweights = {'dense_net': 'densenet_cifar10.h5',
                         'netinnet': 'NetInNet_cifar10.h5', 'resnet': 'ResNet_cifar10.h5',
                         'vgg': 'vgg_cifar10.h5'}
    mnistmodelweights = {'mnist': {'deepxplore': 'deepxplore_mnist.hdf5',
                                   'lenet': 'lenet_mnist.h5', 'mlp': 'mlp_mnist.h5'},
                         'fashion_mnist': {'deepxplore': 'deepxplore_fashion_mnist.hdf5',
                                           'lenet': 'lenet_fashion_mnist.h5',
                                           'mlp': 'mlp_fashion_mnist.h5'}}
    tcount = 0
    tncount = 0
    for m in modellist:
        for d in datalist:
            if 'cifar' in d:
                weightspaths = cifarmodelweights
            else:
                weightspaths = mnistmodelweights[d]
            bestModelName = wpath + weightspaths[m]
            (x_train, _), (x_test, y_test), (_, _, num_class) = datama.getData(d)
            if m == 'mlp':
                x_train = x_train.reshape(x_train.shape[0], -1)
                x_test = x_test.reshape(x_test.shape[0], -1)
            bestModelName = wpath + weightspaths[m]
            if m == 'mlp':
                x_train = x_train.reshape(x_train.shape[0], -1)
            model = helper.load_model(m, bestModelName, x_train.shape[1:], 10, isDrop=False)


            data = np.load('./adv_data/slectedTest300ImgsIdx_%s_%s' % (m, d)).item()
            selectedIndex = data.get("idx")
            ly_test = np.argmax(y_test, axis=1)[..., np.newaxis]
            cx_train, cy_train, cidx = x_test[selectedIndex], ly_test[selectedIndex], selectedIndex

            for a in adv:
                if not os.path.isdir('m_%s_%s_%s'%(a, m,d)):
                    os.mkdir('m_%s_%s_%s'%(a,m,d))
                for j in tqdm(np.arange(300)):
                    ground = cy_train[j]
                    def dowork():
                        count = 0
                        ncount = 0
                        for i in np.arange(5)[::-1]:
                            path = "%s_%s_%s/%d/%d"%(a, m,d,j,i)
                            svaedpath = "m_%s_%s_%s/%d.npy" % (a,m, d, j)
                            if os.path.isdir(path):
                                    if len(os.listdir(path))!=0:
                                           files = os.listdir(path)
                                           if 'best.npy' in files:
                                               #files.remove('best.npy')
                                               finalimg = np.load(os.path.join(path, 'best.npy')).item()
                                               itr = finalimg.get('itr')
                                               sf = natsort.natsorted(files)
                                               total_images = []
                                               def dow():
                                                   n = 0
                                                   c = 0
                                                   for f in sf:
                                                       name = int(f.split('.')[0])
                                                       fpath = os.path.join(path, f)
                                                       images  = np.load(fpath)
                                                       prey = model.predict(images, batch_size=1)
                                                       labels = np.argmax(prey, axis=1)
                                                       c += np.sum(labels==ground)
                                                       n += np.sum(labels!=ground)
                                                       return c, n
                                                       #if itr > name:
                                                           # print(images.shape)
                                                       #    total_images.extend(list(images))
                                                       #else:
                                                       #    total_images.extend(list(images))
                                                       #    break
                                               dc, dn  = dow()
                                               count += dc
                                               ncount += dn
                                               #total_images.append(finalimg.get('img'))
                                               #np.save(svaedpath, total_images)
                                               break
                                           else:
                                               continue
                            else:
                                    continue

                            return count, ncount


                    dc, dn = dowork()
                    tcount += dc
                    tncount += dn

    df = pd.DataFrame(data = {'c':tncount, 'nc': tncount})
    df.to_csv("count.csv")

def summaryFGSM(modellist =['mlp','lenet','deepxplore'], datalist = ['mnist','fashion_mnist'], adv=['FGSM']):
    #modellist =['mlp','lenet','deepxplore', 'netinnet','vgg']
    #datalist = ['mnist','fashion_mnist']
    for m in modellist:
        for d in datalist:
            for a in adv:
                if not os.path.isdir('m_%s_%s_%s'%(a, m,d)):
                    os.mkdir('m_%s_%s_%s'%(a,m,d))
                for j in tqdm(np.arange(300)):
                    def dowork():
                            path = "%s_%s_%s/%d"%(a, m,d,j)
                            svaedpath = "m_%s_%s_%s/%d.npy" % (a,m, d, j)
                            if os.path.isdir(path):
                                    if len(os.listdir(path))!=0:
                                           files = os.listdir(path)
                                           if 'found.npy' in files:
                                               files.remove('found.npy')
                                               finalimg = np.load(os.path.join(path, 'found.npy')).item()
                                               itr = finalimg.get('itr')
                                               sf = natsort.natsorted(files)
                                               total_images = []
                                               for f in sf:
                                                    name = int(f.split('.')[0])
                                                    fpath = os.path.join(path, f)
                                                    images  = np.load(fpath)
                                                    if itr>name:
                                                        #print(images.shape)
                                                        total_images.extend(list(images))
                                                    else:
                                                        total_images.extend(list(images))
                                                        break
                                               total_images.append(finalimg.get('img'))
                                               np.save(svaedpath, total_images)



                    dowork()
if __name__ == '__main__':
    summaryCW(modellist=['mlp'], datalist=['mnist', 'fashion_mnist'])
    #summaryCW(modellist=['lenet', 'deepxplore'], datalist=['mnist', 'fashion_mnist'])
    #summaryCW(modellist=['vgg'], datalist=['cifar10'])
    #summaryCW(modellist=['netinnet'], datalist=['cifar10'])

    summaryFGSM(modellist=['mlp'], datalist=['mnist', 'fashion_mnist'])
    #summaryFGSM(modellist=['lenet', 'deepxplore'], datalist=['mnist', 'fashion_mnist'])
    #summaryFGSM(modellist=['vgg'], datalist=['cifar10'])
    #summaryFGSM(modellist=['netinnet'], datalist=['cifar10'])