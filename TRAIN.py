import glob
#import cv2 #pip install opencv-python
import numpy as np
from PIL import Image  #pip install Pillow
#import matplotlib.pyplot as plt  # plt 用于显示图片
import tensorflow as tf  #注意！tensorflow版本与cuda和cudnn版本有指定的对应要求。https://www.tensorflow.org/install/source#gpu
#解决 COULD NOT LOAD DYNAMIC LIBRARY ‘CUDART64_101.DLL‘ 或 TENSORFLOW 2.3.0 CUDA 10.0 问题 #https://www.freesion.com/article/85351028122/
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
import random
# 字符串操作
import re
from tqdm import tqdm
from tensorflow.keras import Sequential, Model, preprocessing
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Activation, UpSampling2D, Input, Flatten,Dense,add
from tensorflow.keras.layers import Reshape
#分配GPU资源 分配了百分之三十的显存 如果没有显卡可能会慢一些 但不影响程序运行
# config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # 仅在需要时分配显存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # 异常处理
        print(e)

class SpatialGate(tf.keras.Sequential):
    def __init__(self, channel):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.spatial = self.basic_conv(2, 1, kernel_size, stride=1, padding='same', gelu=False)
        self.dw1 = tf.keras.Sequential([
            self.basic_conv(channel, channel, 5, stride=1, dilation_rate=2, padding='same', groups=channel),
            self.basic_conv(channel, channel, 7, stride=1, dilation_rate=3, padding='same', groups=channel)
        ])
        self.dw2 = self.basic_conv(channel, channel, kernel_size, stride=1, padding='same', groups=channel)

    def call(self, x):
        out = tf.concat([tf.reduce_max(x, axis=3, keepdims=True), tf.reduce_mean(x, axis=3, keepdims=True)], axis=3)
        out = self.spatial(out)
        out = tf.tile(out,[1,1,1,512])
        out = self.dw1(x) * out + self.dw2(x)
        return out

    def basic_conv(self, in_planes, out_planes, kernel_size, stride=1, padding='valid', dilation_rate=1, groups=1, gelu=False, bn=False, bias=True):
        layers_list = tf.keras.Sequential()
        layers_list.add(tf.keras.layers.Conv2D(out_planes, kernel_size=kernel_size, strides=stride, padding=padding, dilation_rate=dilation_rate, groups=groups, use_bias=bias))
        if bn:
            layers_list.add(tf.keras.layers.BatchNormalization(momentum=0.01, epsilon=1e-5))
        if gelu:
            layers_list.add(tf.keras.layers.Activation('gelu'))
        return layers_list
class LocalAttention(tf.keras.layers.Layer):
    def __init__(self, channel):
        super(LocalAttention, self).__init__()
        self.channel = channel
        self.sig = tf.keras.layers.Activation('sigmoid')
        self.a = tf.Variable(tf.zeros((1, 1, channel)))
        self.b = tf.Variable(tf.ones((1, 1, channel)))
    def call(self, x):
        out = x - tf.reduce_mean(x, axis=(1, 2), keepdims=True)
        return self.a * out * x + self.b * x

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

class ContextEncoder():
    def __init__(self):
        #图像的行和列
        self.img_rows = 256
        self.img_cols = 256
        #掩膜的高和宽
        self.mask_height = 64
        self.mask_width = 64
        #RGB图像通道数为3
        self.channels = 3

        self.sum_classes = 1
        #将图像的大小和掩膜的大小赋值给两个变量
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.missing_shape = (self.mask_height, self.mask_width, self.channels)
        #优化器，觉得梯度下降情况：参数（学习率，beta1代指你更相信前几次的梯度还是本次的梯度）
        optimizer = Adam(0.0002, 0.5)
        #构建GAN网路的生成器
        self.generator = self.build_generator()
        #构建和编译判别器
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.discriminator.trainable = False
        masked_img = Input(shape=self.img_shape)
        gen_missing = self.generator(masked_img)
        valid = self.discriminator(gen_missing)
        self.combined = Model(masked_img, [gen_missing, valid])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
                              loss_weights=[0.999, 0.001],
                              optimizer=optimizer)
        self.gen_loss=Model(masked_img,gen_missing)
        self.gen_loss.compile(loss=['mse'],
                              loss_weights=[0.999],
                              optimizer=optimizer)



    def build_generator(self):
        # 输入为256*256*3的 被空白膜遮挡的整张图片
        inputs = Input(shape=self.img_shape)
        block_1_output = inputs

        # 编码器第一次卷积
        x = Conv2D(16, kernel_size=4, input_shape=self.img_shape, padding="same")(inputs)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        # ——> 输出256*256*16 的特征图
        block_after_1_output = x

        # 编码器第二次卷积
        x = Conv2D(64, kernel_size=4, strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        # ——> 输出128*128*64 的特征图
        block_2_output = x

        # 编码器第三次卷积
        x = Conv2D(64, kernel_size=4, strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        # ——> 输出64*64*64 的特征图
        block_final_3_out = x

        # 编码器第四次卷积
        x = Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        # ——> 输出32*32*128 的特征图
        block_final_2_out = x

        # 编码器第五次卷积
        x = Conv2D(256, kernel_size=4, strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        # ——> 输出16*16*256 的特征图
        block_final_1_out = x

        # 编码器第六次卷积
        x = Conv2D(512, kernel_size=4, strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        # ——> 编码器最终输出8*8*512 的特征图

        # 进入中间层
        spatial_gate = SpatialGate(channel=512)
        x = spatial_gate.call(x)
        local_attention = LocalAttention(channel=512)
        x = local_attention.call(x)
        shape_before_flatten = tf.keras.backend.int_shape(x)[1:]
        x = Flatten()(x)
        x = Dense(512)(x)  # 调整维度以匹配Transformer的期望输入
        x = Reshape((32, 16))(x)  # 假设Transformer的序列长度为32，维度为16
        # 引入Transformer层
        transformer_block = TransformerEncoderLayer(embed_dim=16, num_heads=4, ff_dim=64)
        x = transformer_block(x)
        # 将输出调整回卷积层的形状
        x = Flatten()(x)
        x = Dense(np.prod(shape_before_flatten))(x)
        x = Reshape(shape_before_flatten)(x)

        # 进入解码器 ：主要由上采样+卷积
        # 解码器第一次上采样+卷积
        x = UpSampling2D()(x)
        x = Conv2D(256, kernel_size=4, padding="same")(x)
        x = Activation('relu')(x)
        x = BatchNormalization(1)(x)
        # 上采样+卷积层+激活层+批量归一化层完成  输出16*16*256
        x = add([x, block_final_1_out])

        # 解码器第二次上采样+卷积
        x = UpSampling2D()(x)
        x = Conv2D(128, kernel_size=4, padding="same")(x)
        x = Activation('relu')(x)
        x = BatchNormalization(1)(x)
        # 上采样+卷积层+激活层+批量归一化层完成  输出32*32*128
        x = add([x, block_final_2_out])

        # 解码器第三次上采样+卷积
        x = UpSampling2D()(x)
        x = Conv2D(64, kernel_size=4, padding="same")(x)
        x = Activation('relu')(x)
        x = BatchNormalization(1)(x)
        # 上采样+卷积层+激活层+批量归一化层完成  输出64*64*64
        x = add([x, block_final_3_out])

        # 解码器第四次上采样+卷积
        x = UpSampling2D()(x)
        x = Conv2D(64, kernel_size=4, padding="same")(x)
        x = Activation('relu')(x)
        x = BatchNormalization(1)(x)
        # 上采样+卷积层+激活层+批量归一化层完成  输出128*128*64
        x = add([x, block_2_output])

        # 解码器第五次上采样+卷积
        x = UpSampling2D()(x)
        x = Conv2D(16, kernel_size=4, padding="same")(x)
        x = Activation('relu')(x)
        x = BatchNormalization(1)(x)
        # 上采样+卷积层+激活层+批量归一化层完成  输出256*256*16
        x = add([x, block_after_1_output])

        # 解码器输出（通过卷积将输出结果输出为256x256x3的大小）
        x = Conv2D(self.channels, kernel_size=2, padding='same')(x)
        x = Activation('tanh')(x)
        # 输出256*256*3
        # 在解码器的输出部分再和原图叠加一下
        x = add([x, block_1_output])

        ###################到此，输入的激光干扰图像经过了一次编码器和解码器的运算#################
        outputs = x
        model = Model(inputs, outputs, name='encoder_decoder_resnet')
        model.summary()  # 展示模型参数和模型尺寸
        return model

    def build_discriminator(self):

        model = Sequential()

        # 输入256*256*3
        model.add(Conv2D(64, kernel_size=4, strides=2, input_shape=self.img_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # 128*128*64

        model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # 64*64*128

        model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # 32*32*128

        model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # 16*16*128

        model.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # 8*8*256

        model.add(Conv2D(512, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # 4*4*512

        model.add(Flatten())
        # 16384

        model.add(Dense(1, activation='sigmoid'))
        model.build()
        model.summary()

        img = Input(self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train_list(self, list_input,list_folder_path,interfering_image_foder_path,epochs=100, batch_size=16, sample_interval=50):

        X_train = []

        print("---g---b---t-------G---------")
        list=list_input
        # list = random.sample(list, 900)
        for l in list:
            try:
                im = Image.open(l)
                # im.show()
                im = im.resize((256, 256))
                # r , g , b = im.split()
                # im = Image.merge("RGB" , (r,g,b))
                # print(type(im))
                im = np.array(im).astype(np.float32)
                # print(type(im))
                im = np.reshape(im, [-1, 256 * 256 * 3])
                # print(type(im))
                im = (im - (255 / 2.0)) / 255
                batch_xs = np.reshape(im, [-1, 256, 256, 3])

                X_train.append(batch_xs)
            except:
                print("except--step")
                pass


        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, len(X_train) - 1, batch_size)
            # print("idx:",idx)
            imgs = []
            list_masked=[]
            for i in idx:
                imgs.append((X_train[i]))
                list_masked.append(list[i])
            # 定义原missing_parts为随机选出的原始图像
            missing_parts = imgs

            list2 = []
            for i in list_masked:
                p = i.lstrip(list_folder_path)
                p = interfering_image_foder_path + p
                list2.append(p)

            id_list = []
            id_new=-1
            X_train2=[]
            for l in list2:
                try:
                    im = Image.open(l)
                    #print("step1--finshed")
                    # im.show()
                    im = im.resize((256, 256))
                    im = np.array(im).astype(np.float32)
                    im = np.reshape(im, [-1, 256 * 256 * 3])
                    im = (im - (255 / 2.0)) / 255
                    #print("step5--finshed")
                    batch_xs = np.reshape(im, [-1, 256, 256, 3])
                    #print("step6--finshed")

                    X_train2.append(batch_xs)
                    id_new = id_new + 1
                    id_list.append(id_new)
                    #print("step7--finshed")
                except:
                    print("except--2")
                    pass
            # print('missing_parts', len(missing_parts), "---------list2加载完成----------")
            # 要保missing_parts 和masked_imgs长度一样，masked_imgs可能比missing_parts短
            missing_parts_new = []
            for i in id_list:
                missing_parts_new.append(missing_parts[i])
            missing_parts = missing_parts_new
            masked_imgs = X_train2
            # masked_imgs代表了遮挡的batch个图像
            # missing_parts代表了丢失的batch个图像块
            # print(masked_imgs.shape)
            # gen_missing=np.empty_like(masked_imgs)
            gen_missing = []
            x = -1;
            for i in X_train2:
                x = x + 1
                result = self.generator.predict(i)
                gen_missing.append(result)

            g_losses = []
            for i in range(len(masked_imgs)):
                g_loss = self.gen_loss.train_on_batch(masked_imgs[i], missing_parts[i])
                g_losses.append(g_loss)

            # 计算本次epoch的平均生成器损失
            avg_g_loss = np.mean(g_losses)

            # 打印损失信息
            print(f"Epoch: {epoch + 1}/{epochs}, Generator Loss: {avg_g_loss}")

if __name__ == '__main__':
    print('HE has no partner')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    context_encoder = ContextEncoder()

    list = glob.glob('D:/data/1celeba/训练/原/*.png')
    list_folder_path = 'D:/data/1celeba/训练/原'
    interfering_image_foder_path = 'D:/data/1celeba/训练/干扰'
    for i in tqdm(range(100)):
        context_encoder.train_list(list, list_folder_path, interfering_image_foder_path)

        # 检查当前迭代次数 i，如果 i+1 是 10 的倍数，则保存模型
        # 这里使用 i+1 是因为 range(100) 生成的 i 从 0 开始，所以第一次迭代是 i=0
        if (i + 1) % 10 == 0:
            context_encoder.generator.save(f'1原1dsm-ts256-cel1-{i + 1}.h5') # 保存模型


