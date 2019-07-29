import utils.load_data as datama
import utils.attacker as attacker
import utils.tools as helper
import logging
import numpy as np
import foolbox
from keras.utils import to_categorical
import argparse
import os

def compute_shuffle_idx(size):
    (x_train, y_train), (_, _), (_, _, num_class) = datama.getData('mnist')
    idx_mnist = np.arange(x_train.shape[0])
    np.random.shuffle(idx_mnist)

    (x_train, y_train), (_, _), (_, _, num_class) = datama.getData('cifar10')
    idx_cifar10 = np.arange(x_train.shape[0])
    np.random.shuffle(idx_cifar10)
    data = {'mnist':idx_mnist[:size], 'fashion_mnist':idx_mnist[:size], 'cifar10': idx_cifar10[:size]}
    if not os.path.isdir('./adv_data/'):
        os.mkdir('./adv_data/')
    np.save('./adv_data/data_index.npy', data)
    return idx_mnist[:size], idx_cifar10[:size]



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-m','--model', type=str,default='mlp')
    ap.add_argument('-d', '--dataset', type=str, default='mnist')
    #ap.add_argument('-s', '--size', type=int, default=10000)
    ap.add_argument('-a','--attacker', nargs='+')
    args = vars(ap.parse_args())

    #size = args['size']
    dataname = args['dataset']
    attack_methods = args['attacker']
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    file_handler = logging.FileHandler('./attack_logs/%s_%s_attack.log'%(args['model'], dataname))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Model %s, Dataset %s"%(args['model'], dataname))
    #model_cifar10 = ['dense_net', 'netinnet', 'resnet', 'vgg']
    path = './cluster_results/model/'
    cifarmodelweights = {'dense_net': 'densenet_cifar10.h5',
                    'netinnet':'NetInNet_cifar10.h5', 'resnet':'ResNet_cifar10.h5',
                    'vgg':'vgg_cifar10.h5'}
    mnistmodelweights = {'mnist':{'deepxplore':'deepxplore_mnist.hdf5',
                                  'lenet':'lenet_mnist.h5', 'mlp':'mlp_mnist.h5'},
                         'fashion_mnist':{'deepxplore':'deepxplore_fashion_mnist.hdf5',
                                          'lenet':'lenet_fashion_mnist.h5',
                                          'mlp':'mlp_fashion_mnist.h5'}}
    #model_mnist = ['mlp','deepxplore', 'lenet']
    datalist = ['mnist', 'fashion_mnist','cifar10']

    if 'cifar' in dataname:
        bounds = (0, 255.)
        weightspaths = cifarmodelweights
    else:
        bounds = (0, 1.0)
        weightspaths = mnistmodelweights[dataname]

    name = args['model']

    (x_train, _), (x_test, y_test), (_, _, num_class) = datama.getData(dataname)
    if name == 'mlp':
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    bestModelName = path + weightspaths[name]
    if name == 'mlp':
        x_train = x_train.reshape(x_train.shape[0], -1)
        # x_test = x_test.reshape(x_test.shape[0], -1)
    model = helper.load_model(name, bestModelName, x_train.shape[1:], num_class, isDrop=False)

    del x_train
    y_test = np.argmax(y_test, axis=1)
    if not os.path.isfile('./adv_data/slectedTest300ImgsIdx_%s_%s'%(name, dataname)):
            y = model.predict(x_test)
            plabels = np.argmax(y, axis=1)
            b = plabels == y_test
            #idx = np.arange(x_test.shape[0])
            idx = np.nonzero(b)[0]
            np.random.shuffle(idx)
            cx_train, cy_train, selectedIndex = x_test[idx[:300]], y_test[idx[:300]], idx[:300]
            cidx = selectedIndex
            np.save('./adv_data/slectedTest300ImgsIdx_%s_%s'%(name, dataname), {"idx":selectedIndex})
    else:
            data = np.load('./adv_data/slectedTest300ImgsIdx_%s_%s'%(name, dataname)).item()
            selectedIndex = data.get("idx")
            (_, _), (x_test, y_test), (_, _, num_class) = datama.getData(dataname)
            y_test = np.argmax(y_test, axis=1)[..., np.newaxis]
            cx_train, cy_train, cidx = x_test[selectedIndex], y_test[selectedIndex], selectedIndex




    if "FGSM" in attack_methods:
         print("FGSM Attacking")
         if not os.path.isdir('./FGSM_%s_%s/' % (name, dataname)):
             os.mkdir('./FGSM_%s_%s/' % (name, dataname))
         x_fgsm, s_fgsm = attacker.attackFGSM(model, cx_train, cy_train, bounds,svaeMedianImages='./FGSM_%s_%s/'%(name, dataname))
         #idx_fgsm = cidx[s_fgsm]
         score = model.evaluate(x_fgsm, to_categorical(cy_train[s_fgsm],num_classes=10),verbose=0, batch_size=1)
         logger.info("FGSM {}, {}".format(name, score))
         data = {'fgsm': {'x_adv': x_fgsm, 'y_adv': cy_train[s_fgsm], 'idx': selectedIndex[s_fgsm]}}
         np.save('./adv_data/%s_%s_fgsm.np' % (name, dataname), data)
         del x_fgsm,data

    # if "DF" in attack_methods:
    #     print("DF Attacking")
    #     x_df, s_df = attacker.attackDeepFool(model, cx_train, cy_train, bounds)
    #     idx_df = cidx[s_df]
    #     score = model.evaluate(x_df, to_categorical(cy_train[s_df]),verbose=0, batch_size=1)
    #     logger.info("DF {}, {}".format(name, score))
    #     data = { 'df': {'x_adv': x_df, 'y_adv': cy_train[s_df], 'idx': selectedIndex[idx_df]}}
    #     np.save('./adv_data/%s_%s_df.np' % (name, dataname), data)
    #     del x_df,idx_df, data
    #
    # if "BIM" in attack_methods:
    #     print("BIM Attacking")
    #     x_bim, s_bim = attacker.attackBIM(model, cx_train, cy_train, bounds)
    #     idx_bim = cidx[s_bim]
    #     score = model.evaluate(x_bim, to_categorical(cy_train[s_bim]),verbose=0, batch_size=1)
    #     logger.info("BIM {}, {}".format(name, score))
    #     data = {'bim': {'x_adv': x_bim, 'y_adv': cy_train[s_bim], 'idx': selectedIndex[idx_bim]}}
    #     np.save('./adv_data/%s_%s_bim.np' % (name, dataname), data)
    #     del x_bim,s_bim,data
    #
    # if 'JSMA' in attack_methods:
    #     print("JSMA Attacking")
    #     x_jsma, s_jsma = attacker.attackJSMA(model, cx_train, cy_train, bounds)
    #     idx_jsma = cidx[s_jsma]
    #     score = model.evaluate(x_jsma, to_categorical(cy_train[s_jsma]),verbose=0, batch_size=1)
    #     logger.info("JSMA {}, {}".format(name, score))
    #     data = {'jsma': {'x_adv': x_jsma, 'y_adv': cy_train[s_jsma], 'idx': selectedIndex[idx_jsma]}}
    #     np.save('./adv_data/%s_%s_jsma.np'%(name, dataname), data)
    #     del x_jsma,idx_jsma,data

    if 'CW' in attack_methods:
        print("CW Attacking")
        if not os.path.isdir('./CW_%s_%s/' % (name, dataname)):
            os.mkdir('./CW_%s_%s/' % (name, dataname))
        x_cw, s_cw = attacker.attackCWl2(model, cx_train, cy_train, bounds,svaeMedianImages='./CW_%s_%s/'%(name, dataname))
        #idx_cw = cidx[s_cw]
        score = model.evaluate(x_cw, to_categorical(cy_train[s_cw],num_classes=10), verbose=0, batch_size=1)
        logger.info("CW {}, {}".format(name, score))
        data = {'cw':{'x_adv':x_cw, 'y_adv':cy_train[s_cw], 'idx': selectedIndex[s_cw]}}
        np.save('./adv_data/%s_%s_cw.np'%(name, dataname), data)



