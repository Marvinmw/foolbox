import numpy as np
import natsort
import os
from tqdm import tqdm
def summaryCW(modellist =['mlp','lenet','deepxplore'], datalist = ['mnist','fashion_mnist'], adv=['CW']):
    #modellist =['mlp','lenet','deepxplore', 'netinnet','vgg']
    #datalist = ['mnist','fashion_mnist']
    for m in modellist:
        for d in datalist:
            for a in adv:
                if not os.path.isdir('m_%s_%s_%s'%(a, m,d)):
                    os.mkdir('m_%s_%s_%s'%(a,m,d))
                for j in tqdm(np.arange(300)):
                    def dowork():
                        for i in np.arange(5)[::-1]:
                            path = "%s_%s_%s/%d/%d"%(a, m,d,j,i)
                            svaedpath = "m_%s_%s_%s/%d.npy" % (a,m, d, j)
                            if os.path.isdir(path):
                                    if len(os.listdir(path))!=0:
                                           files = os.listdir(path)
                                           if 'best.npy' in files:
                                               files.remove('best.npy')
                                               sf = natsort.natsorted(files)
                                               total_images = []
                                               for f in sf:
                                                   fpath = os.path.join(path, f)
                                                   images  = np.load(fpath)
                                                   #print(images.shape)
                                                   total_images.extend(list(images))
                                               finalimg = np.load(os.path.join(path, 'best.npy'))
                                               total_images.append(finalimg.get('img'))
                                               np.save(svaedpath, total_images)
                                               break
                                           else:
                                               continue
                            else:
                                    continue


                    dowork()

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
                            path = "%s_%s_%s/%d/%d"%(a, m,d,j)
                            svaedpath = "m_%s_%s_%s/%d.npy" % (a,m, d, j)
                            if os.path.isdir(path):
                                    if len(os.listdir(path))!=0:
                                           files = os.listdir(path)
                                           if 'found.npy' in files:
                                               files.remove('found.npy')
                                               sf = natsort.natsorted(files)
                                               total_images = []
                                               for f in sf:
                                                   fpath = os.path.join(path, f)
                                                   images  = np.load(fpath)
                                                   #print(images.shape)
                                                   total_images.extend(list(images))
                                               finalimg = np.load(os.path.join(path, 'found.npy'))
                                               total_images.append(finalimg.get('img'))
                                               np.save(svaedpath, total_images)



                    dowork()
if __name__ == '__main__':
    summaryCW(modellist=['lenet', 'deepxplore'], datalist=['mnist', 'fashion_mnist'])
    summaryCW(modellist=['vgg'], datalist=['cifar10'])
    summaryCW(modellist=['netinnet'], datalist=['cifar10'])

    summaryFGSM(modellist=['lenet', 'deepxplore'], datalist=['mnist', 'fashion_mnist'])
    summaryFGSM(modellist=['vgg'], datalist=['cifar10'])
    summaryFGSM(modellist=['netinnet'], datalist=['cifar10'])