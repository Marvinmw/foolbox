import numpy as np
import natsort
import os
def summaryCW(modellist =['mlp','lenet','deepxplore'], datalist = ['mnist','fashion_mnist']):
    #modellist =['mlp','lenet','deepxplore', 'netinnet','vgg']
    #datalist = ['mnist','fashion_mnist']
    for m in modellist:
        for d in datalist:
            if not os.path.isdir('m_CW_%s_%s'%(m,d)):
                os.mkdir('m_CW_%s_%s'%(m,d))
            for j in np.arange(300):

                def dowork():
                    for i in np.arange(5)[::-1]:
                        path = "CW_%s_%s/%d/%d"%(m,d,j,i)
                        svaedpath = "m_CW_%s_%s/%d.npy" % (m, d, j)
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
                                               print(images.shape)
                                               total_images.extend(list(images))
                                               np.save(svaedpath, total_images)
                                           break
                                       else:
                                           continue
                        else:
                                continue


                dowork()


if __name__ == '__main__':
    summaryCW(modellist=['mlp', 'lenet', 'deepxplore'], datalist=['mnist', 'fashion_mnist'])
    summaryCW(modellist=['vgg'], datalist=['cifar10'])