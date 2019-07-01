from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D,Lambda
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers
from keras import backend as K

# set GPU memory
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)



def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
    return x_train, x_test


def scheduler(epoch):
    if epoch < 81:
        return 0.1
    if epoch < 122:
        return 0.01
    return 0.001


def residual_network(img_input, classes_num=10, stack_n=5, drop=False, droprate=0.2):
    count = 0
    def residual_block(x, o_filters,count, increase=False):
        stride = (1, 1)
        if increase:
            stride = (2, 2)

        o1 = Activation('relu',  name='l'+str(count))(BatchNormalization(momentum=0.9, epsilon=1e-5,  name='l'+str(count+1))(x))
        count = count + 2
        conv_1 = Conv2D(o_filters, kernel_size=(3, 3), strides=stride, padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay),  name='l'+str(count))(o1)
        count = count + 1
        o2 = Activation('relu',  name='l'+str(count))(BatchNormalization(momentum=0.9, epsilon=1e-5,  name='l'+str(count+1))(conv_1))
        count = count + 2
        conv_2 = Conv2D(o_filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay),  name='l'+str(count))(o2)
        count = count + 1
        if increase:
            projection = Conv2D(o_filters, kernel_size=(1, 1), strides=(2, 2), padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay),  name='l'+str(count))(o1)
            count = count + 1
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
        return block, count

    # build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay),  name='l'+str(count))(img_input)
    count = count + 1
    # input: 32x32x16 output: 32x32x16
    for i in range(stack_n):
        x,count = residual_block(x, 16,count, False)
    if drop:
        x = Lambda(lambda x: K.dropout(x, level=droprate))(x)

    # input: 32x32x16 output: 16x16x32
    x , count= residual_block(x, 32,count, True)
    for i in range(1, stack_n):
        x, count = residual_block(x, 32,count, False)

    if drop:
        x = Lambda(lambda x: K.dropout(x, level=droprate))(x)


    # input: 16x16x32 output: 8x8x64
    x, count = residual_block(x, 64, count, True)
    for i in range(1, stack_n):
        x , count= residual_block(x, 64, count, False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5,  name='l'+str(count))(x)
    count = count +1
    x = Activation('relu',  name='l'+str(count))(x)
    count = count + 1
    x = GlobalAveragePooling2D( name='l'+str(count))(x)
    count = count + 1
    # input: 64 output: 10
    x = Dense(classes_num, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay), name='l'+str(count))(x)
    return x

import utils.load_data as datama
from  keras import callbacks

stack_n            = 5
layers             = 6 * stack_n + 2
batch_size         = 128

weight_decay       = 1e-4
def train(dataset='cifar10', **kwargs):
    print("========================================")
    print("MODEL: Residual Network ({:2d} layers)".format(6 * stack_n + 2))
    print("BATCH SIZE: {:3d}".format(batch_size))
    print("WEIGHT DECAY: {:.4f}".format(weight_decay))
    epochs = kwargs['epochs'] if 'epochs' in kwargs else 300
    print("EPOCHS: {:3d}".format(epochs))
    if not 'data' in kwargs:
        (x_train, y_train), (x_test, y_test), (img_rows, img_cols, num_classes) = datama.getData(dataset)
    else:
        ((x_train, y_train), (x_test, y_test), num_classes) = kwargs['data']
        img_rows, img_cols = x_train.shape[1], x_train.shape[2]
    # color preprocessing
    #x_train, x_test = color_preprocessing(x_train, x_test)
    print("== DONE! ==\n== BUILD MODEL... ==")
    # build network
    img_input = Input(shape=x_train.shape[1:])
    output = residual_network(img_input, num_classes, stack_n)
    resnet = Model(img_input, output)
    print(resnet.summary())
    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set callback
    if not 'logfile' in kwargs:
        csvlog = callbacks.CSVLogger("./log/ResNet_" + dataset + ".log", separator=',', append=False)
    else:
        csvlog = callbacks.CSVLogger(kwargs['logfile'], separator=',', append=False)

    if not 'bestModelfile' in kwargs:
            checkPoint = callbacks.ModelCheckpoint('./model/ResNet_' + dataset + ".h5",
                                                   save_best_only=True,monitor="val_acc",verbose=1)
    else:
        checkPoint = ModelCheckpoint(kwargs['bestModelfile'], monitor="val_acc", save_best_only=True, verbose=1)
    cbks = [LearningRateScheduler(scheduler),csvlog,checkPoint]

    # dump checkpoint if you need.(add it to cbks)
    # ModelCheckpoint('./checkpoint-{epoch}.h5', save_best_only=False, mode='auto', period=10)

    # set data augmentation
    print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant', cval=0.)

    datagen.fit(x_train)
    iterations = x_train.shape[0] // batch_size
    # start training
    resnet.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                         steps_per_epoch=iterations,
                         epochs=epochs,
                         callbacks=cbks,
                         validation_data=(x_test, y_test))
    #resnet.save('resnet_{:d}_{}.h5'.format(layers,  dataset))
