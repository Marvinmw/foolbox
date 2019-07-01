from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, Activation, AveragePooling2D, GlobalAveragePooling2D, Lambda, concatenate
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers



from keras import backend as K
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


def scheduler(epoch):
    if epoch < 150:
        return 0.1
    if epoch < 225:
        return 0.01
    return 0.001



def densenet(img_input,classes_num, drop=False, droprate=0.2):
    count = 0
    def conv(x, out_filters, k_size,count):
        return Conv2D(filters=out_filters,
                      kernel_size=k_size,
                      strides=(1,1),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay),
                      use_bias=False,  name='l'+str(count))(x), count+1

    def dense_layer(x,count):
        return Dense(units=classes_num,
                     activation='softmax',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(weight_decay), name='l'+str(count))(x), count+1

    def bn_relu(x,count):
        x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='l'+str(count))(x)
        x = Activation('relu', name='l'+str(count+1))(x)
        return x, count+2

    def bottleneck(x, count):
        channels = growth_rate * 4
        x,count = bn_relu(x, count)
        x,count = conv(x, channels, (1,1), count)
        x,count = bn_relu(x, count)
        x,count = conv(x, growth_rate, (3,3),count)
        return x,count

    def single(x, count):
        x,count = bn_relu(x, count)
        x,count = conv(x, growth_rate, (3,3), count)
        return x, count

    def transition(x, inchannels,count):
        outchannels = int(inchannels * compression)
        x,count = bn_relu(x,count)
        x,count = conv(x, outchannels, (1,1),count)
        x = AveragePooling2D((2,2), strides=(2, 2),  name='l'+str(count))(x)
        count = count + 1
        return x, outchannels,count

    def dense_block(x,blocks,nchannels,count):
        concat = x
        for i in range(blocks):
            x,count = bottleneck(concat,count)
            concat = concatenate([x,concat], axis=-1)
            nchannels += growth_rate
        return concat, nchannels,count
    nblocks = (depth - 4) // 6
    nchannels = growth_rate * 2
    x,count = conv(img_input, nchannels, (3,3),count)
    x, nchannels,count = dense_block(x,nblocks,nchannels,count)
    if drop:
        x = Lambda(lambda x: K.dropout(x, level=droprate))(x)
    x, nchannels,count = transition(x,nchannels,count)
    if drop:
        x = Lambda(lambda x: K.dropout(x, level=droprate))(x)
    x, nchannels,count = dense_block(x,nblocks,nchannels,count)
    if drop:
        x = Lambda(lambda x: K.dropout(x, level=droprate))(x)
    x, nchannels,count = transition(x,nchannels,count)
    if drop:
        x = Lambda(lambda x: K.dropout(x, level=droprate))(x)
    x, nchannels,count = dense_block(x,nblocks,nchannels,count)
    x,count = bn_relu(x,count)
    x = GlobalAveragePooling2D( name='l'+str(count))(x)
    count = count + 1
    x,count = dense_layer(x,count)
    return x

from keras import callbacks
import utils.load_data as datama

growth_rate        = 12
depth              = 100
compression        = 0.5
batch_size         = 64        # 64 or 32 or other
weight_decay       = 1e-4
def train(dataset, epochs=300, **kwargs):
    epochs = epochs
    if not 'data' in kwargs:
        (x_train, y_train), (x_test, y_test), (img_rows, img_cols, num_classes) = datama.getData(dataset)
    else:
        ((x_train, y_train), (x_test, y_test),num_classes) = kwargs['data']
        img_rows, img_cols = x_train.shape[1],x_train.shape[2]
    # build network
    img_input = Input(shape=x_train.shape[1:])
    output = densenet(img_input, num_classes)
    model = Model(img_input, output)
    print(model.summary())
    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # set callback
    #tb_cb = TensorBoard(log_dir='./log/densenet_'+dataset+".log", histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    if not 'logfile' in kwargs:
        csvlog = callbacks.CSVLogger("./log/densenet_" + dataset + ".log", separator=',', append=False)
    else:
        csvlog = callbacks.CSVLogger(kwargs['logfile'], separator=',', append=False)
    if not 'bestModelfile' in kwargs:
        ckpt = ModelCheckpoint('./model/densenet_'+dataset+".h5", save_best_only=True, monitor="val_acc", verbose=1)
    else:
        ckpt = ModelCheckpoint(kwargs['bestModelfile'], save_best_only=True, monitor="val_acc", verbose=1)
    cbks = [change_lr, csvlog, ckpt]

    # set data augmentation
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.125, height_shift_range=0.125,
                                 fill_mode='constant', cval=0.)

    datagen.fit(x_train)
    iterations = x_train.shape[0]//epochs
    # start training
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=iterations,
                        epochs=epochs, callbacks=cbks, validation_data=(x_test, y_test))
