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
parser.add_argument('--epochs', default=10, type=int, help="number of epochs for training")
parser.add_argument('--batch_size', default=16, type=int, help="batch size")
parser.add_argument('--learning_rate', default=0.001, type=float, help="initial learning rate")
args = parser.parse_args()

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


def augment_image(img, gt_image):
    return img, gt_image


def data_generator(file_list, batch_size, image_shape=[80,265], augment_images=False):
    idx = 0
    background_color = np.array([255, 0, 0])
    scipy_image_shape = (image_shape[1], image_shape[0])

    while True:
        images = []
        gt_images = []

        for i in range(batch_size):
            img_file_name = file_list[idx]
            gt_file_name = get_label_file_name(img_file_name)

            image = scipy.misc.imresize(scipy.misc.imread(img_file_name), scipy_image_shape)
            gt_image = scipy.misc.imresize(scipy.misc.imread(gt_file_name), scipy_image_shape)

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
    vgg_input_tensor = sess.graph.get_tensor_by_name('image_input:0')
    vgg_keep_prob_tensor = sess.graph.get_tensor_by_name('keep_prob:0')
    vgg_layer3_out_tensor = sess.graph.get_tensor_by_name('layer3_out:0')
    vgg_layer4_out_tensor = sess.graph.get_tensor_by_name('layer4_out:0')
    vgg_layer7_out_tensor = sess.graph.get_tensor_by_name('layer7_out:0')

    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor

tests.test_load_vgg(load_vgg, tf)


def custom_init(shape, dtype=tf.float32, partition_info=None, seed=0):
    return tf.random_normal(shape, dtype=dtype, seed=seed)


def encoder(input):
    return tf.layers.conv2d(input, NUM_CLASSES, 1, (1, 1))


def decoder(input, factor):
    return tf.layers.conv2d_transpose(
	input, NUM_CLASSES, factor, strides=(factor//2,factor//2), padding='same') 


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
    val_generator, num_val,
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
    num_val_batches = (num_val // batch_size) + (num_val % batch_size > 0)

    def feed_dict(generator):
        images, labels = next(generator)
        return {input_image: images, correct_label: labels, keep_prob:0.5}

    def eval_epoch(generator, num_batches, train_op=None):
        loss = 0.0
        pb = ProgressBar(num_batches)
        for b in range(num_batches):
            args = [cross_entropy_loss]
            if train_op is not None:
                args.append(train_op)
            result = sess.run(args, feed_dict=feed_dict(generator))
            loss += result[0] / num_batches
            pb(b, message="%.4f" % loss)
        return loss

    best_validation_loss = float('inf')
 
    for i in range(epochs):
        training_loss = eval_epoch(train_generator, num_train_batches, train_op)
        validation_loss = eval_epoch(val_generator, num_val_batches)        

        if validation_loss < best_validation_loss:
            print("Validation loss decreased to from %.4f to %.4f. Saving model." 
                    % (best_validation_loss, validation_loss))
            best_validation_loss = validation_loss
        else:
            print("Validation loss: %.4f" % validation_loss)

#tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    #image_shape = (80, 265)
    #image_shape = (40, 132)
    data_dir = './data'
    runs_dir = './runs'
    batch_size = 4
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    train, val = train_validation_split("data/data_road", val_split=0.2)
    train_generator = data_generator(train, batch_size, image_shape=image_shape, augment_images=True)
    val_generator = data_generator(val, batch_size, image_shape=image_shape, augment_images=False)


    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_tensor, keep_prob, layer3_tensor, layer4_tensor, layer7_tensor = load_vgg(sess, './data/vgg')
        out_layer = layers(layer3_tensor, layer4_tensor, layer7_tensor, num_classes)
        labels_tensor = tf.placeholder(tf.float32, shape=[None, None, None, num_classes])
        logits, train_op, loss = optimize(out_layer, labels_tensor, num_classes)
        learning_rate = tf.placeholder(tf.float32) 

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())


        # TODO: Train NN using the train_nn function
        train_nn(
            sess,
            args.epochs, args.batch_size,
            train_generator, len(train),
            val_generator, len(val),
            train_op, loss,
            input_tensor, labels_tensor,
            keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
