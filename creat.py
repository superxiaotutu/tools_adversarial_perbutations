import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import numpy as np
import os
import PIL
import matplotlib.pyplot as plt
import matplotlib.image as image
import glob
from scipy.misc import imread, imresize

IMG_SIZE = 299

adv_placehoder = tf.Variable(tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]))
X_placehoder = tf.placeholder(tf.float32, [1, IMG_SIZE, IMG_SIZE, 3])
X_test_placehoder = tf.placeholder(tf.float32, [1, IMG_SIZE, IMG_SIZE, 3])


sess = tf.InteractiveSession()

# globel arg
total_target = 3
alpha = 0.3
epsilon = 10
Epoch_NUM = 3


# load data
img_set = [] #数据集合
def load_img():
    imgs_list = glob.glob("data/n01440764/*")
    for i_path in imgs_list:
        I = PIL.Image.open(i_path)
        I = I.resize((IMG_SIZE, IMG_SIZE)).crop((0, 0, IMG_SIZE, IMG_SIZE))
        img_set.append(np.asarray(I))



# model A
def inception(image, reuse=tf.AUTO_REUSE):
    preprocessed = tf.multiply(tf.subtract(image, 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, end_point = nets.inception.inception_v3(
            preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:, 1:]  # ignore background class
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point

# def resnet(image, reuse=tf.AUTO_REUSE):
#     preprocessed = tf.multiply(tf.subtract(image, 0.5), 2.0)
#     arg_scope = nets.resnet_v2.resnet_arg_scope(weight_decay=0.0)
#     with slim.arg_scope(arg_scope):
#         logits, end_point = nets.resnet_v2.resnet_v2_50(preprocessed, 1001, is_training=False, reuse=reuse)
#         # logits = logits[:, 1:]  # ignore background class
#         probs = tf.nn.softmax(logits)  # probabilities
#     return logits, probs, end_point

X_log, X_prob, X_end_point = inception(X_placehoder + adv_placehoder)


# creat adv from model A
adv_labels = tf.one_hot(total_target, 1000)
adv_LOSS = tf.nn.softmax_cross_entropy_with_logits(logits=X_log, labels=[adv_labels])
reflush_adv = tf.train.AdamOptimizer(100).minimize(adv_LOSS, var_list=[adv_placehoder])
# grad_of_X = tf.gradients(adv_LOSS, X_placehoder)[0]
# reflush_adv = tf.assign(adv_placehoder, adv_placehoder - alpha * tf.sign(grad_of_X))
# project
projected = tf.assign(adv_placehoder, tf.clip_by_value(adv_placehoder, - epsilon, epsilon))
def creat_adv(origin_img):
    original_label = np.argmax(sess.run(X_prob, feed_dict={X_placehoder: [origin_img]}))
    time = 0
    current_label = np.argmax(sess.run(X_prob, feed_dict={X_placehoder: origin_img + sess.run(adv_placehoder)}))
    while True:
        if current_label == total_target or time == 100:
            break
        current_adv_placehoder = sess.run(adv_placehoder)
        current_label = np.argmax(sess.run(X_prob, feed_dict={X_placehoder: origin_img + current_adv_placehoder}))
        sess.run(reflush_adv, feed_dict={X_placehoder: origin_img + current_adv_placehoder})
        sess.run(projected)
        time += 1
        print("ori:{}, curr:{}".format(original_label, current_label))

_, X_test, _ = inception(X_test_placehoder)
img_label = tf.argmax(tf.squeeze(X_test))



# total tongyon adv
def calculation_util_adv():
    for epoh in range(Epoch_NUM):
        for it, img in enumerate(img_set):
            try:
                creat_adv(img)
                print("next img")
            except ValueError:
                print('black_white img')
                pass
            print(it)
            if it == 1000:
                break
        print("finish one epoch!" + ' ' + str(epoh))

    return 0

def preprocess_image_batch(image_paths, img_size=None, crop_size=None, color_mode="rgb", out=None):
    img_list = []

    for im_path in image_paths:
        img = imread(im_path, mode='RGB')
        if img_size:
            img = imresize(img,img_size)

        img = img.astype('float32')
        # We normalize the colors (in RGB space) with the empirical means on the training set
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        # We permute the colors to get them in the BGR order
        # if color_mode=="bgr":
        #    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]

        if crop_size:
            img = img[(img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2, (img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2, :];

        img_list.append(img)

    try:
        img_batch = np.stack(img_list, axis=0)
    except:
        raise ValueError('when img_size and crop_size are None, images'
                ' in image_paths must have the same shapes.')

    if out is not None and hasattr(out, 'append'):
        out.append(img_batch)
    else:
        return img_batch

def undo_image_avg(img):
    img_copy = np.copy(img)
    img_copy[:, :, 0] = img_copy[:, :, 0] + 123.68
    img_copy[:, :, 1] = img_copy[:, :, 1] + 116.779
    img_copy[:, :, 2] = img_copy[:, :, 2] + 103.939
    return img_copy


restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('InceptionV3/')
]

saver = tf.train.Saver(restore_vars)

if __name__ == '__main__':

    file_perturbation = 'final_output.npy'
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'inception/inception_v3.ckpt')

    if os.path.isfile(file_perturbation) == 0:
        load_img()
        calculation_util_adv()
        final_adv_mask = sess.run(adv_placehoder)
        np.save("final_output.npy", final_adv_mask)
    else:
        X = np.load("final_output.npy")
    #target_fooling_rate = target_fooling_rate_calc(v=pre_v, dataset=X, f=f, target=target)
    #print("")
    #print('TARGET FOOLING RATE = ', target_fooling_rate)

    test_img = image.imread('data/tiger.jpg')

    test_img = preprocess_image_batch(['data/tiger.jpg'], img_size=(299, 299),  color_mode="rgb")

    # Show original and perturbed image
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(undo_image_avg(test_img[0, :, :, :]).astype(dtype='uint8'), interpolation=None)
    #test_img_expand = np.expand_dims(test_img, axis=0)
    str_label_original = sess.run(img_label,feed_dict={X_test_placehoder:test_img})
    plt.title(str_label_original)

    # plt.subplot(1, 2, 2)
    # plt.imshow(undo_image_avg(image_perturbed[0, :, :, :]).astype(dtype='uint8'), interpolation=None)
    # plt.title(str_label_perturbed)

    plt.subplot(1, 2, 2)
    plt.imshow(undo_image_avg((test_img+X)[0, :, :, :]).astype(dtype='uint8'), interpolation=None)
    str_label_adv = sess.run(img_label, feed_dict={X_test_placehoder: test_img+X})
    plt.title(str_label_adv)

    plt.show()

