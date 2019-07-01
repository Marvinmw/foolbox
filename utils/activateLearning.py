import numpy as np
np.random.seed(698686)
import utils.tools as utils
#import utils.SADL as SADL
import utils.sa as sa
def computeVariancescore(model, data, drop_rep = 50, num_class=10):
    '''
    :param model:
    :param data:
    :param drop_rep:
    :return:
    '''
    # X, Y,num_repeat, num_class, model
    (x, y, nb_class) = data
    (result, label, _, _, _, _) = \
            utils.predict(x, y, drop_rep, num_class, model)
    # Sort variance
    var_all_class = np.var(result, axis=0)
    var_mean_all_class = np.mean(var_all_class, axis=1)
    del model, data, x, y
    return var_mean_all_class

def getSamplesByVar(model, remainingData, num, drop_rep=50, **kwargs):
    '''

    :param model:
    :param remainingData:
    :param num:
    :param drop_rep:
    :param drop_rate:
    :param dataset:
    :return:
    '''
    #X, Y,num_repeat, num_class, model
    (x,y) = remainingData
    import keras.backend as K
    import tensorflow as tf

    (result, label, counter, p_rlabel, variance, means) = \
        utils.predict(x, y, drop_rep, kwargs["num_class"], model)
    #Sort variance
    var_all_class = np.var(result, axis=0)
    var_mean_all_class = np.mean(var_all_class, axis=1)
    p, _ = utils.prob_mean(result)

    if kwargs['method'] == 'var':
        ind = np.argsort(var_mean_all_class)[::-1] #Descending order
        #Splitting data
        #x_jointrain = x[ind[:num]]
        #y_jointrain = y[ind[:num]]
        #x_remaining = x[ind[num:]]
        #y_remaining = y[ind[num:]]
        (x_jointrain, y_jointrain), (x_remaining, y_remaining) = \
            selectData(x[ind], y[ind], kwargs["num_each"], kwargs['num_class'])
    elif kwargs['method']== 'varW':
        ormodel = kwargs['orginalModel']
        py = ormodel.predict(x)
        pl = np.argmax(py, axis=1)
        pp = np.squeeze(py[np.arange(len(py)), pl])
        var_mean_all_class = np.squeeze(var_mean_all_class)
        var_mean_all_class = np.divide(var_mean_all_class, pp)
        ind = np.argsort(var_mean_all_class)[::-1]
        (x_jointrain, y_jointrain), (x_remaining, y_remaining) = \
            selectData(x[ind], y[ind], kwargs["num_each"], kwargs['num_class'])
    else :
        nb_bins = 50
        dic, score, p, vbins, pbins= utils.compute2DHistGroup(var_mean_all_class, p, nb_bins)
        res = []
        for i in range(nb_bins)[::-1]:
            vidx, pidx = i, nb_bins-1-i
            for n in range(pidx+1):
                res.extend(dic[nb_bins-1-n][pidx])
            for h in range(pidx):
                res.extend(dic[nb_bins-1-pidx][h])
            if len(res)>=num:
                break
        res = np.asarray(res)
        idx = res[:num]
        idx_left = np.ones(len(x), dtype=bool)
        idx_left[idx] = False
        x_jointrain, y_jointrain = x[idx], y[idx]
        x_remaining, y_remaining = x[idx_left], y[idx_left]

    del result
    del label
    del counter
    del p_rlabel
    del variance
    del means,model
    return (x_jointrain, y_jointrain),(x_remaining, y_remaining),var_mean_all_class

def getSamplesRandom(remainingData, num, **kwargs):
    (x, y) = remainingData
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    #x_jointrain = x[idx[:num]]
    #y_jointrain = y[idx[:num]]
    #x_remaining = x[idx[num:]]
    #y_remaining = y[idx[num:]]
    (x_jointrain, y_jointrain), (x_remaining, y_remaining) = \
        selectData(x[idx], y[idx], kwargs['num_each'], kwargs['num_class'])
    del x
    del y, remainingData
    return (x_jointrain, y_jointrain), (x_remaining, y_remaining), None

#def computeRefTrace(model, x_reference,**kwargs):
 #   '''

    #:param model:
    #:param x_reference:
    #:param y_reference:
    #:param y_pre:
    #:param kwargs:
    #:return:
    #'''
    #correctlabel = y_reference == y_pre
    #print(correctlabel.shape)
    #selectX = ()
    #selectY = ()
    #Sampel reference
    #for i in np.arange(kwargs["num_class"]):
    #    i_labelIdx = np.argwhere(y_reference==i)#np.logical_and(correctlabel, y_reference==i)
    #    assert len(i_labelIdx)!=0,"%d no this labele in the reference data to compute the trace"%i
    #    i_labelIdx = np.squeeze(i_labelIdx)
    #    x_i = x_reference[i_labelIdx]
    #    a = np.arange(len(x_i))
    #    np.random.shuffle(a)
    #    ref_x_i = x_i[a][:kwargs["ref_sample_num"]]
    #    selectX += (ref_x_i, )
    #    ref_y_i = np.repeat(i, len(ref_x_i)).T
    #    selectY += (ref_y_i, )
    #x_refsample = np.concatenate(selectX, axis=0)
    #assert x_refsample.shape[0] == kwargs["num_class"] * kwargs["ref_sample_num"]
    #y_refsample = np.concatenate(selectY, axis=0)
    #assert y_refsample.shape[0] == kwargs["num_class"] * kwargs["ref_sample_num"]

    #trace_ref = SADL.get_ActivationTrace(model,x_reference, kwargs["layers"])# x_refsample, kwargs["layers"])
    #assert trace_ref.shape[0] == kwargs["num_class"] * kwargs["ref_sample_num"]
    #del model, x_reference
    #return trace_ref

def computeDSAscore(model, remainingData, **kwargs):
    '''

    :param model:
    :param remainingData:
    :param kwargs:
    :return:
    '''
    #p = Pool(20)
    import utils.sa as sa


    (x_reference, y_reference, y_pre) = kwargs["xref"]
    print(x_reference.shape)
    print(y_reference.shape)
    print(y_pre.shape)
    (x, y) = remainingData
    #xprey = model.predict_classes(x, verbose=0, batch_size=128)
    #xx = model.predict(x, verbose=0, batch_size=128)
    #xprey = xx.argmax(axis=-1)

    dsascores = sa.fetch_dsa(model, x_reference, x, "candidates", kwargs["layers"], num_classes=10, var_threshold=1e-5,
                             is_classification=True)

    #trace_ref = computeRefTrace(model, x_reference, **kwargs) if not ('reftrace' in kwargs) else kwargs['reftrace']
    #samplesTrace = SADL.get_ActivationTrace(model, x, kwargs["layers"])
    #dsascores = np.zeros(len(x))
    #for i in np.arange(len(x)):
    #    assert  np.sum(xprey[i] == y_pre)>0,"No this label %d."%xprey[i]
    #    if np.sum(xprey[i] == y_pre)>0:
    #        dsascores[i] = SADL.DSA(samplesTrace[i], xprey[i], trace_ref, y_pre)
    #    else:
    #        dsascores[i] = np.finfo(np.float).max
    #dsascores = p.map(dsacore_task, [(samplesTrace[i], xprey[i], trace_ref, y_pre)  for i in np.arange(len(x)) ])
    dsascores = np.asarray(dsascores)

    del x_reference
    del y_reference
    del y_pre
    del x
    del y
    del remainingData, model
    return    dsascores

def getSamplesDSA(model, remainingData, num, **kwargs):
    '''

    :param model:
    :param remainingData:
    :param num:
    :param kwargs:
    :return:
    '''
    (x, y) = remainingData
    dsascores = computeDSAscore(model, remainingData, **kwargs)#np.zeros(len(x))
    idx = np.argsort(dsascores)[::-1]
    #x_jointrain = x[idx[:num]]
    #y_jointrain = y[idx[:num]]
    #x_remaining = x[idx[num:]]
    #y_remaining = y[idx[num:]]
    (x_jointrain, y_jointrain), (x_remaining, y_remaining) = \
        selectData(x[idx], y[idx],  kwargs['num_each'], kwargs['num_class'])

    del x
    del y
    del remainingData, model
    return (x_jointrain, y_jointrain), (x_remaining, y_remaining),dsascores

def computeLSAscore(model, remainingData, **kwargs):
    '''

    :param model:
    :param remainingData:
    :param kwargs:
    :return:
    '''

    (x_reference, y_reference, y_pre) = kwargs["xref"]
    (x, _) = remainingData
    #xypre = model.predict_classes(x, batch_size=128, verbose=0)
    #xx = model.predict(x, verbose=0, batch_size=128)
    #xypre = xx.argmax(axis=-1)
    #trace_ref = computeRefTrace(model, x_reference, **kwargs)  if not ('reftrace' in kwargs) else kwargs['reftrace']
    #samplesTrace = SADL.get_ActivationTrace(model, x, kwargs["layers"])

    lsascores = sa.fetch_dsa(model, x_reference, x, "candidates", kwargs["layers"], num_classes=10, var_threshold=1e-5,
                             is_classification=True)

    # Train KDE
    #kde = {} if not ('kde' in kwargs) else kwargs['kde']
    #lsascores = np.zeros(len(x))
    #trace_var = np.var(trace_ref, axis=1)
    #del_features = np.nonzero(trace_var<1e-5)[0]
    #trace_ref = np.delete(trace_ref, del_features, axis=1)
    #samplesTrace = np.delete(samplesTrace, del_features, axis=1)

    #for i in np.arange(kwargs["num_class"]):
    #    if np.sum(y_pre == i)>0:
    #        kde[i] = SADL.trainKDE(trace_ref[y_pre == i])  if not ('kde' in kwargs) else kde[i]
    #    if np.sum(xypre == i) != 0 and (i in kde):
    #        lsascores[xypre == i] = -kde[i].score_samples(samplesTrace[xypre == i])
    #    else:
    #        if np.sum(xypre == i) != 0:
    #            lsascores[xypre == i] = np.finfo(np.float).max
    del  x_reference,  y_reference  , y_pre, x, model
    return lsascores

def getSamplesLSA(model, remainingData, num, **kwargs):
    '''

    :param model:
    :param remainingData:
    :param num:
    :param kwargs:
    :return:
    '''
    (x, y) = remainingData
    lsascores = computeLSAscore(model, remainingData, **kwargs)

    idx = np.argsort(lsascores)[::-1]
    (x_jointrain, y_jointrain), (x_remaining, y_remaining) = \
        selectData(x[idx], y[idx],  kwargs['num_each'], kwargs['num_class'])
    del x, y, remainingData, model
    return (x_jointrain, y_jointrain), (x_remaining, y_remaining),lsascores


def computeSilhouttescore(model, remainingData, **kwargs):
    '''

    :param model:
    :param remainingData:
    :param kwargs:
    :return:
    '''

    (x_reference, y_reference, y_pre) = kwargs["xref"]
    (x, _) = remainingData

    sihoutete = sa.fetch_sihoutete(model, x_reference, x, "candidates", kwargs["layers"], num_classes=10, var_threshold=1e-5,
                             is_classification=True)
    del  x_reference,  y_reference  , y_pre, x, model
    return sihoutete

def getSamplesSilhoutte(model, remainingData, num, **kwargs):
    '''

    :param model:
    :param remainingData:
    :param num:
    :param kwargs:
    :return:
    '''
    (x, y) = remainingData
    silhoutte = computeSilhouttescore(model, remainingData, **kwargs)

    idx = np.argsort(silhoutte)
    (x_jointrain, y_jointrain), (x_remaining, y_remaining) = \
        selectData(x[idx], y[idx],  kwargs['num_each'], kwargs['num_class'])
    del x, y, remainingData, model
    return (x_jointrain, y_jointrain), (x_remaining, y_remaining),silhoutte



def selectData(x, y, num_each, nb_classes):
    newy = ()
    newx = ()
    leftx = ()
    lefty = ()

    def f(x, y):
        x  = np.concatenate(x, axis=0)
        #print(x.shape)
        y = np.concatenate(y, axis=0)
        #print(y.shape)
        a = np.arange(len(x))
        np.random.shuffle(a)
        return (x[a], y[a])

    for i in num_each:
        if num_each[i] == 0:
            continue
        idx = (y == i)
        newy += (y[idx][:num_each[i]], )
        newx += (x[idx][:num_each[i]], )
        leftx += (x[idx][num_each[i]:],)
        lefty += (y[idx][num_each[i]:],)
    return f(newx, newy), f(leftx, lefty)

# #Test
def computeKLScore(model, remainingData, drop_rep=50, **kwargs):
    '''

        :param model:
        :param remainingData:
        :param num:
        :param drop_rep:
        :param drop_rate:
        :param dataset:
        :return:
        '''
    # X, Y,num_repeat, num_class, model
    (x, y) = remainingData
    (result, label, _, _, _, _) = \
        utils.predict(x, y, drop_rep, kwargs["num_class"], model)

    # Sort divergence
    kl, var_hist = utils.computeKL(result, kwargs["num_class"], label)
    del result, label, model
    return kl,var_hist

def getKLDiverge(model, remainingData, num, drop_rep=50, **kwargs):
    '''

    :param model:
    :param remainingData:
    :param num:
    :param drop_rep:
    :param drop_rate:
    :param dataset:
    :return:
    '''
    #X, Y,num_repeat, num_class, model
    (x,y) = remainingData

    (result, label, counter, p_rlabel, variance, means) = \
        utils.predict(x, y, drop_rep, kwargs["num_class"], model)
    #Sort divergence
    kl, var_hist = utils.computeKL(result, kwargs["num_class"], label)
    if kwargs["method"]=='KL':
        ind = np.argsort(kl)
        (x_jointrain, y_jointrain), (x_remaining, y_remaining) = \
            selectData(x[ind], y[ind], kwargs['num_each'], kwargs['num_class'])
    else:
        nb_bins = 50
        p,_ = utils.prob_mean(result)
        dic, score, p, vbins, pbins = utils.compute2DHistGroup(kl, p, nb_bins)
        res = []
        for i in range(nb_bins):
            vidx, pidx = i, i
            for n in range(pidx + 1):
                res.extend(dic[n][pidx])
            for h in range(vidx):
                res.extend(dic[vidx][h])
            if len(res) >= num:
                break
        res = np.asarray(res)
        idx = res[:num]
        idx_left = np.ones(len(x), dtype=bool)
        idx_left[idx] = False
        x_jointrain, y_jointrain = x[idx], y[idx]
        x_remaining, y_remaining = x[idx_left], y[idx_left]

    del result, label, counter, p_rlabel, variance, means,model
    return (x_jointrain, y_jointrain),(x_remaining, y_remaining),kl

def getLabelHist(model, remainingData, num, drop_rep=50, **kwargs):
    '''

    :param model:
    :param remainingData:
    :param num:
    :param drop_rep:
    :param drop_rate:
    :param dataset:
    :return:
    '''
    #X, Y,num_repeat, num_class, model
    (x,y) = remainingData
    import keras.backend as K
    import tensorflow as tf

    (result, label, counter, p_rlabel, variance, means) = \
        utils.predict(x, y, drop_rep, kwargs["num_class"], model)
    #Sort divergence
    _, var_hist = utils.computeKL(result, kwargs["num_class"], label)
    ind = np.argsort(var_hist)
    (x_jointrain, y_jointrain), (x_remaining, y_remaining) = \
        selectData(x[ind], y[ind], kwargs['num_each'], kwargs['num_class'])
    del result, label, counter, p_rlabel, variance, means,model
    return (x_jointrain, y_jointrain),(x_remaining, y_remaining),var_hist

def getNeuroCover(model, remainingData, num, **kwargs):
    (x, y) = remainingData
    ncComputor = kwargs['ncComputor']
    if kwargs['method'] == 'NC':
        nc_score = ncComputor.batch_nc(x)
    if kwargs['method'] == 'KMNC':
        (nc_score,_,_) = ncComputor.batch_kmnc(x)
    if kwargs['method'] == 'BNC':
        (_,nc_score, _) = ncComputor.batch_kmnc(x)
    if kwargs['method'] == 'SANC':
        (_,_, nc_score) = ncComputor.batch_kmnc(x)

    if kwargs['method'] == 'DiffNC':
        nc_score = ncComputor.batch_diffScore( x)

    idx = np.argsort(nc_score)[::-1]
    if np.any(nc_score==0):
        non_zero = np.nonzero(nc_score)[0]
        usedIdx = idx[:non_zero.size]
        leftIdx = idx[non_zero.size:]
        np.random.shuffle(leftIdx)
        idx = np.concatenate([usedIdx, leftIdx])

    (x_jointrain, y_jointrain), (x_remaining, y_remaining) = \
        selectData(x[idx], y[idx], kwargs['num_each'], kwargs['num_class'])
    del x, y, remainingData, model
    return (x_jointrain, y_jointrain), (x_remaining, y_remaining), nc_score


def getSamplesByP(model, remainingData, num, **kwargs):
    (x, y) = remainingData
    py = model.predict(x)
    pl = np.argmax(py, axis=1)
    maxp = np.squeeze(py[np.arange(len(py)), pl])
    idx = np.argsort(maxp)
    (x_jointrain, y_jointrain), (x_remaining, y_remaining) = \
        selectData(x[idx], y[idx], kwargs['num_each'], kwargs['num_class'])
    del x, y, remainingData, model
    return (x_jointrain, y_jointrain), (x_remaining, y_remaining), maxp

def getSamples(model, remainingData, num, drop_rep=50, method='var', **kwargs):
    if method == 'P':
        return getSamplesByP(model, remainingData, num, **kwargs)

    if 'var' in method:
        #model, remainingData, num, drop_rep=50, **kwargs
        return getSamplesByVar(model, remainingData, num,
                               drop_rep=drop_rep,method=method, **kwargs)

    if method=='random':
        #remainingData, num, **kwargs
        return getSamplesRandom(remainingData, num, **kwargs)

    if method=='DSA':
        #model, remainingData, num, **kwargs
        return  getSamplesDSA(model, remainingData, num, **kwargs)

    if method=='LSA':
        #model, remainingData, num, **kwargs
        return  getSamplesLSA(model, remainingData, num, **kwargs)
    if method== 'silhoutte':
        return getSamplesSilhoutte(model, remainingData, num, **kwargs)
    if 'KL' in method:
        return getKLDiverge(model, remainingData, num, drop_rep=drop_rep,method=method, **kwargs)

    if method=='Hist':
        return getLabelHist(model, remainingData, num, drop_rep=drop_rep, **kwargs)

    if 'NC' in method:
        return getNeuroCover(model, remainingData, num, method=method, **kwargs)


