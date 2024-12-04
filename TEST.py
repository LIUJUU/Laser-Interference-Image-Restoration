import time
from tensorflow import keras
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import os

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
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

def calculate_metrics(image_true, image_pred):
    mse = mean_squared_error(image_true, image_pred)
    psnr = peak_signal_noise_ratio(image_true, image_pred, data_range=255)
    ssim = structural_similarity(image_true, image_pred, data_range=255, multichannel=True, win_size=3)
    return mse, psnr, ssim

def show(path, model, comparison_folder):
    im = Image.open(path)
    im = np.array(im).astype(np.float32)
    im = np.reshape(im, [-1, 256 * 256 * 3])
    im = (im - (255 / 2.0)) / 255
    im = np.reshape(im, [-1, 256, 256, 3])

    origin_img = np.reshape(im, [-1, 256 * 256 * 3])
    origin_img = origin_img * 255 + 255 / 2.0
    origin_img = np.reshape(origin_img, [256, 256, 3]).astype(int)
    plt.subplot(1, 2, 1)
    plt.imshow(origin_img)

    result = model.predict(im)
    result = np.reshape(result, [-1, 256 * 256 * 3])
    result = result * 255 + 255 / 2.0
    result = np.clip(result, 0, 255)
    result = np.reshape(result, [256, 256, 3]).astype(np.uint8)
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    # plt.show()

    file_name = os.path.basename(path)
    comparison_path = os.path.join(comparison_folder, file_name)
    comparison_img = Image.open(comparison_path)
    comparison_img = np.array(comparison_img).astype(np.float32)

    mse, psnr, ssim = calculate_metrics(comparison_img, result)
    return mse, psnr, ssim

if __name__ == '__main__':
    load_save_model = keras.models.load_model(r'C:\pyobj\2\5000-dsm-ts256-bud光斑改-100.h5', custom_objects={'TransformerEncoderLayer': TransformerEncoderLayer})
    folder_dict = {
        1: (r'C:\data\1celeba\测试\干扰', r'C:\data\1celeba\测试\原'),
        2: (r'D:\data\3Places365_256x256\building_facade\测试\干扰', r'D:\data\3Places365_256x256\building_facade\测试\原'),
        3: (r'D:\data\3Places365_256x256\airfield\测试\干扰', r'D:\data\3Places365_256x256\airfield\测试\原'),
        4: (r'D:\data\3Places365_256x256\auto_showroom\测试\干扰', r'D:\data\3Places365_256x256\auto_showroom\测试\原'),
        5: (r'D:\data\3Places365_256x256\tower\测试\干扰', r'D:\data\3Places365_256x256\tower\测试\原'),
        6: (r'C:\pyobj\real\干扰', r'C:\pyobj\real\原'),
    }
    index = 2
    path_folder_1, path_folder_2 = folder_dict[index]

    image_paths_1 = glob.glob(path_folder_1 + '/*.jpg')
    image_paths_1 += glob.glob(path_folder_1 + '/*.png')

    total_mse, total_psnr, total_ssim = 0, 0, 0
    num_images = len(image_paths_1)

    start_time = time.time()  # Start time

    for idx, path in enumerate(image_paths_1):
        mse, psnr, ssim = show(path, load_save_model, path_folder_2)
        total_mse += mse
        total_psnr += psnr
        total_ssim += ssim
        print(f'Processing {idx + 1}/{num_images} images')

    end_time = time.time()  # End time

    avg_mse = total_mse / num_images
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    total_time = end_time - start_time
    avg_time_per_image = total_time / num_images

    print(f'Average MSE: {avg_mse}')
    print(f'Average PSNR: {avg_psnr}')
    print(f'Average SSIM: {avg_ssim}')
    print(f'Total processing time: {total_time:.3f} seconds')  # Print total processing time
    print(f'Average time per image: {avg_time_per_image:.3f} seconds')  # Print average time per image
