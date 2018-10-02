#! -*- coding: utf-8 -*-

import numpy as np
import glob
import imageio
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.datasets import cifar10


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255 - 0.5
x_test = x_test.astype('float32') / 255 - 0.5
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)
img_dim = x_train.shape[1]


z_dim = 256 # 隐变量维度
alpha = 0.5 # 全局互信息的loss比重
beta = 1.5 # 局部互信息的loss比重
gamma = 0.01 # 先验分布的loss比重


# 编码器（卷积与最大池化）

x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in

for i in range(3):
    x = Conv2D(z_dim / 2**(2-i),
               kernel_size=(3,3),
               padding='SAME')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D((2, 2))(x)

feature_map = x # 截断到这里，认为到这里是feature_map（局部特征）
feature_map_encoder = Model(x_in, x)


for i in range(2):
    x = Conv2D(z_dim,
               kernel_size=(3,3),
               padding='SAME')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

x = GlobalMaxPooling2D()(x) # 全局特征

z_mean = Dense(z_dim)(x) # 均值，也就是最终输出的编码
z_log_var = Dense(z_dim)(x) # 方差，这里都是模仿VAE的


encoder = Model(x_in, z_mean) # 总的编码器就是输出z_mean


# 重参数技巧
def sampling(args):
    z_mean, z_log_var = args
    u = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * u


# 重参数层，相当于给输入加入噪声
z_samples = Lambda(sampling)([z_mean, z_log_var])
prior_kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))


# shuffle层，打乱第一个轴
def shuffling(x):
    idxs = K.arange(0, K.shape(x)[0])
    idxs = K.tf.random_shuffle(idxs)
    return K.gather(x, idxs)


# 与随机采样的特征拼接（全局）
z_shuffle = Lambda(shuffling)(z_samples)
z_z_1 = Concatenate()([z_samples, z_samples])
z_z_2 = Concatenate()([z_samples, z_shuffle])

# 与随机采样的特征拼接（局部）
feature_map_shuffle = Lambda(shuffling)(feature_map)
z_samples_repeat = RepeatVector(4 * 4)(z_samples)
z_samples_map = Reshape((4, 4, z_dim))(z_samples_repeat)
z_f_1 = Concatenate()([z_samples_map, feature_map])
z_f_2 = Concatenate()([z_samples_map, feature_map_shuffle])


# 全局判别器
z_in = Input(shape=(z_dim*2,))
z = z_in
z = Dense(z_dim, activation='relu')(z)
z = Dense(z_dim, activation='relu')(z)
z = Dense(z_dim, activation='relu')(z)
z = Dense(1, activation='sigmoid')(z)

GlobalDiscriminator = Model(z_in, z)

z_z_1_scores = GlobalDiscriminator(z_z_1)
z_z_2_scores = GlobalDiscriminator(z_z_2)
global_info_loss = - K.mean(K.log(z_z_1_scores + 1e-6) + K.log(1 - z_z_2_scores + 1e-6))


# 局部判别器
z_in = Input(shape=(None, None, z_dim*2))
z = z_in
z = Dense(z_dim, activation='relu')(z)
z = Dense(z_dim, activation='relu')(z)
z = Dense(z_dim, activation='relu')(z)
z = Dense(1, activation='sigmoid')(z)

LocalDiscriminator = Model(z_in, z)

z_f_1_scores = LocalDiscriminator(z_f_1)
z_f_2_scores = LocalDiscriminator(z_f_2)
local_info_loss = - K.mean(K.log(z_f_1_scores + 1e-6) + K.log(1 - z_f_2_scores + 1e-6))

# 用来训练的模型
model_train = Model(x_in, [z_z_1_scores, z_z_2_scores, z_f_1_scores, z_f_2_scores])
model_train.add_loss(alpha * global_info_loss + beta * local_info_loss + gamma * prior_kl_loss)
model_train.compile(optimizer=Adam(1e-3))

model_train.fit(x_train, epochs=50, batch_size=64)


# 输出编码器的特征
zs = encoder.predict(x_train, verbose=True)
zs.mean() # 查看均值（简单观察先验分布有没有达到效果）
zs.std() # 查看方差（简单观察先验分布有没有达到效果）


# 随机选一张图片，输出最相近的图片
# 可以选用欧氏距离或者cos值
def sample_knn(path):
    n = 10
    topn = 10
    figure1 = np.zeros((img_dim*n, img_dim*topn, 3))
    figure2 = np.zeros((img_dim*n, img_dim*topn, 3))
    zs_ = zs / (zs**2).sum(1, keepdims=True)**0.5
    for i in range(n):
        one = np.random.choice(len(x_train))
        idxs = ((zs**2).sum(1) + (zs[one]**2).sum() - 2 * np.dot(zs, zs[one])).argsort()[:topn]
        for j,k in enumerate(idxs):
            digit = x_train[k]
            figure1[i*img_dim: (i+1)*img_dim,
                   j*img_dim: (j+1)*img_dim] = digit
        idxs = np.dot(zs_, zs_[one]).argsort()[-n:][::-1]
        for j,k in enumerate(idxs):
            digit = x_train[k]
            figure2[i*img_dim: (i+1)*img_dim,
                   j*img_dim: (j+1)*img_dim] = digit
    figure1 = (figure1 + 1) / 2 * 255
    figure1 = np.clip(figure1, 0, 255)
    figure2 = (figure2 + 1) / 2 * 255
    figure2 = np.clip(figure2, 0, 255)
    imageio.imwrite(path+'_l2.png', figure1)
    imageio.imwrite(path+'_cos.png', figure2)


sample_knn('test')
