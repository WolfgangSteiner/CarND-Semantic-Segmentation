import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import argparse
from progress_bar import ProgressBar
import glob
import random
import numpy as np
import scipy

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=25, type=int, help="number of epochs for training")
parser.add_argument('--batch_size', default=1, type=int, help="batch size")
parser.add_argument('--learning_rate', default=0.0001, type=float, help="initial learning rate")
parser.add_argument('--freeze_vgg',  action="store_true", default=False, help="freeze vgg during training")
parser.add_argument('--regularize', action="store_true", default=False, help="use l2 regularization")
parser.add_argument('--keep_prob', default=0.75, type=float, help="keep probability")
args = parser.parse_args()

print(args)



################################################################################
# 			       GLOBAL CONSTANTS                                #
################################################################################
NUM_CLASSES=2


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def train_validation_split(data_dir, val_split=0.25):
    img_path = os.path.join(data_dir, "training", "image_2")
    img_files = glob.glob(os.path.join(img_path, "*.png"))
    print("Found %d training images." % len(img_files))
    random.shuffle(img_files)
    num_training = int(len(img_files) * (1.0 - val_split))
    return img_files[0:num_training], img_files[num_training:]


def get_label_file_name(img_file_name):
    label_file_name = img_file_name.replace("_0", "_road_0").replace("image_2", "gt_image_2")
    assert os.path.exists(label_file_name)
    return label_file_name


def augment_image(image, gt_image):
    if random.random() > 0.5:
        image = np.fliplr(image)
        gt_image = np.fliplr(gt_image)
    return image, gt_image


def data_generator(file_list, batch_size, image_shape=[80,265], augment_images=False):
    idx = 0
    background_color = np.array([255, 0, 0])

    while True:
        images = []
        gt_images = []

        for i in range(batch_size):
            img_file_name = file_list[idx]
            gt_file_name = get_label_file_name(img_file_name)

            image = scipy.misc.imresize(scipy.misc.imread(img_file_name), image_shape)
            gt_image = scipy.misc.imresize(scipy.misc.imread(gt_file_name), image_shape)

            gt_bg = np.all(gt_image == background_color, axis=2)
            gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
            gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

            if augment_images:
                image, gt_image = augment_image(image, gt_image)

            images.append(image)
            gt_images.append(gt_image)
            idx = (idx + 1) % len(file_list)

        yield np.array(images), np.array(gt_images)


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    graph = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    vgg_input = sess.graph.get_tensor_by_name('image_input:0')
    vgg_keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
    vgg_layer3_out = sess.graph.get_tensor_by_name('layer3_out:0')
    vgg_layer4_out = sess.graph.get_tensor_by_name('layer4_out:0')
    vgg_layer7_out = sess.graph.get_tensor_by_name('layer7_out:0')
    
    if args.freeze_vgg:
        vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)
        vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
        vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)

    return vgg_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out


if not args.freeze_vgg:
    tests.test_load_vgg(load_vgg, tf)


def custom_init(stddev=0.01):
   return None
   # Using the following initialization method, as suggested in the last review,
   # results in no usable segmenation at all!	
   # return tf.truncated_normal_initializer(stddev)


def encoder(input):
    regularizer = tf.contrib.layers.l2_regularizer(1e-3) if args.regularize else None
    return tf.layers.conv2d(input, NUM_CLASSES, 1, (1, 1),
        kernel_regularizer=regularizer,
        kernel_initializer=custom_init())


def decoder(input, factor):
    regularizer = tf.contrib.layers.l2_regularizer(1e-3) if args.regularize else None
    return tf.layers.conv2d_transpose(
	input, NUM_CLASSES, factor, strides=(factor//2,factor//2), 
	padding='same',
        kernel_regularizer=regularizer,
        kernel_initializer=custom_init())


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    conv3 = encoder(vgg_layer3_out)
    conv4 = encoder(vgg_layer4_out)
    conv7 = encoder(vgg_layer7_out)
    deconv1 = decoder(conv7, 4)
    sum1 = tf.add(conv4, deconv1)
    deconv2 = decoder(sum1, 4)
    sum2 = tf.add(conv3, deconv2)
    return decoder(sum2, 16) 

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return logits, train_op, loss

tests.test_optimize(optimize)


def train_nn(
    sess,
    epochs, batch_size,
    train_generator, num_train,
    train_op, cross_entropy_loss,
    input_image, correct_label,
    keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param train_generator: Generator for training images
    :param val_generator: Generator for validation images
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    current_training_loss = tf.placeholder(tf.float32)
    current_training_loss_summary = tf.summary.scalar(
        'current_training_loss', current_training_loss)

    num_train_batches = (num_train // batch_size) + (num_train % batch_size > 0)

    def feed_dict(generator):
        images, labels = next(generator)
        return {input_image: images, correct_label: labels, keep_prob:args.keep_prob}

    def eval_epoch(generator, epoch, num_batches, train_op=None):
        loss = 0.0
        pb = ProgressBar(num_batches)
        for b in range(num_batches):
            args = [cross_entropy_loss]
            if train_op is not None:
                args.append(train_op)
            result = sess.run(args, feed_dict=feed_dict(generator))
            loss += (result[0] / num_batches)
            pb(b, head="Epoch %03d:" % epoch, message="LOSS: %.4f" % loss)
        return loss

    for epoch in range(epochs):
        training_loss = eval_epoch(train_generator, epoch, num_train_batches, train_op)


#tests.test_train_nn(train_nn)


def run():
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    helper.maybe_download_pretrained_vgg(data_dir)

    train, _ = train_validation_split("data/data_road", val_split=0.0)
    train_generator = data_generator(train, args.batch_size, image_shape=image_shape, augment_images=True)

    with tf.Session() as sess:
        vgg_path = os.path.join(data_dir, 'vgg')

        input_tensor, keep_prob, layer3_tensor, layer4_tensor, layer7_tensor = load_vgg(sess, './data/vgg')
        out_layer = layers(layer3_tensor, layer4_tensor, layer7_tensor, NUM_CLASSES)
        labels_tensor = tf.placeholder(tf.float32, shape=[None, None, None, NUM_CLASSES])
        logits, train_op, loss = optimize(out_layer, labels_tensor, NUM_CLASSES)
        learning_rate = tf.placeholder(tf.float32) 

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_nn(
            sess,
            args.epochs, args.batch_size,
            train_generator, len(train),
            train_op, loss,
            input_tensor, labels_tensor,
            keep_prob, learning_rate)

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_tensor)


if __name__ == '__main__':
    run()
