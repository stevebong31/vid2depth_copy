import tensorflow as tf
import util, data_gen
import numpy as np

slim = tf.contrib.slim

DISP_SCALING = 10
MIN_DISP = 0.01
WEIGHT_REG = 0.0005
EGOMOTION_VEC_SIZE = 6

    
def _resize_like(inputs, ref):
    if tf.shape(inputs)[1] == tf.shape(ref)[1] and tf.shape(inputs)[2] == tf.shape(ref)[2]:
        return inputs
    else:
        return tf.image.resize_nearest_neighbor(inputs, [tf.shape(ref)[1], tf.shape(ref)[2]])    

def disp_net(target_image, is_training = True):
    # predict inverse of depth form a single images.
    batch_norm_params = {'is_training' : is_training}
    inputs = target_image
    
    with tf.variable_scope('depth_net') as sc:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params,
                            weights_regularizer=slim.l2_regularizer(WEIGHT_REG),
                            activation_fn=tf.nn.relu):
            cnv1 = slim.conv2d(inputs, 32, [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1, 32, [7, 7], stride=1, scope='cnv1b')
            cnv2 = slim.conv2d(cnv1b, 64, [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2, 64, [5, 5], stride=1, scope='cnv2b')

            cnv3 = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3, 128, [3, 3], stride=1, scope='cnv3b')
            cnv4 = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4, 256, [3, 3], stride=1, scope='cnv4b')
            cnv5 = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5, 512, [3, 3], stride=1, scope='cnv5b')
            cnv6 = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6, 512, [3, 3], stride=1, scope='cnv6b')
            cnv7 = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7, 512, [3, 3], stride=1, scope='cnv7b')

            up7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling.
            up7 = _resize_like(up7, cnv6b)
            i7_in = tf.concat([up7, cnv6b], axis=3)
            icnv7 = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            up6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            up6 = _resize_like(up6, cnv5b)
            i6_in = tf.concat([up6, cnv5b], axis=3)
            icnv6 = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            up5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            up5 = _resize_like(up5, cnv4b)
            i5_in = tf.concat([up5, cnv4b], axis=3)
            icnv5 = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            up4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
            i4_in = tf.concat([up4, cnv3b], axis=3)
            icnv4 = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            disp4 = (slim.conv2d(icnv4, 1, [3, 3], stride=1, activation_fn=tf.sigmoid,
                               normalizer_fn = None, scope='disp4')
                   * DISP_SCALING + MIN_DISP)
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(tf.shape(target_image)[1] / 4), np.int(tf.shape(target_image)[2] / 4)])

            up3 = slim.conv2d_transpose(icnv4, 64, [3, 3], stride=2, scope='upcnv3')
            i3_in = tf.concat([up3, cnv2b, disp4_up], axis=3)
            icnv3 = slim.conv2d(i3_in, 64, [3, 3], stride=1, scope='icnv3')
            disp3 = (slim.conv2d(icnv3, 1, [3, 3], stride=1, activation_fn=tf.sigmoid,
                               normalizer_fn = None, scope='disp3')
                   * DISP_SCALING + MIN_DISP)
            disp3_up = tf.image.resize_bilinear(disp3, [np.int(tf.shape(target_image)[1] / 2), np.int(tf.shape(target_image)[2] / 2)])

            up2 = slim.conv2d_transpose(icnv3, 32, [3, 3], stride=2, scope='upcnv2')
            i2_in = tf.concat([up2, cnv1b, disp3_up], axis=3)
            icnv2 = slim.conv2d(i2_in, 32, [3, 3], stride=1, scope='icnv2')
            disp2 = (slim.conv2d(icnv2, 1, [3, 3], stride=1, activation_fn=tf.sigmoid,
                               normalizer_fn = None, scope='disp2')
                   * DISP_SCALING + MIN_DISP)
            disp2_up = tf.image.resize_bilinear(disp2, [tf.shape(target_image)[1], tf.shape(target_image)[2]])

            up1 = slim.conv2d_transpose(icnv2, 16, [3, 3], stride=2, scope='upcnv1')
            i1_in = tf.concat([up1, disp2_up], axis=3)
            icnv1 = slim.conv2d(i1_in, 16, [3, 3], stride=1, scope='icnv1')
            disp1 = (slim.conv2d(icnv1, 1, [3, 3], stride=1, activation_fn=tf.sigmoid,
                               normalizer_fn = None, scope='disp1')
                   * DISP_SCALING + MIN_DISP)

            return [disp1, disp2, disp3, disp4]

def egomotion_net(image_stack, is_training =True):
    # Predict ego-motion vectors from a stack of frames.
    #    Network inputs will be [1, 2, 3]
    #    Network outputs will be [1 -> 2, 2 -> 3]
    # Returns:
    #    Egomotion vectors with shape [B, seq_length - 1, 6].    
    batch_norm_params = {'is_training': is_training}
    num_egomotion_vecs = seq_length - 1
    with tf.variable_scope('pose_exp_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        normalizer_fn = slim.batch_norm if FLAGS.use_bn else None
        normalizer_params = batch_norm_params if FLAGS.use_bn else None
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=normalizer_fn,
                            weights_regularizer=slim.l2_regularizer(WEIGHT_REG),
                            normalizer_params=normalizer_params,
                            activation_fn=tf.nn.relu):
            cnv1 = slim.conv2d(image_stack, 16, [7, 7], stride=2, scope='cnv1')
            cnv2 = slim.conv2d(cnv1, 32, [5, 5], stride=2, scope='cnv2')
            cnv3 = slim.conv2d(cnv2, 64, [3, 3], stride=2, scope='cnv3')
            cnv4 = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
            cnv5 = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')

          # Ego-motion specific layers
            with tf.variable_scope('pose'):
                cnv6 = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
                cnv7 = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                pred_channels = EGOMOTION_VEC_SIZE * num_egomotion_vecs
                egomotion_pred = slim.conv2d(cnv7,
                                             pred_channels,
                                             [1, 1],
                                             scope='pred',
                                             stride=1,
                                             normalizer_fn=None,
                                             activation_fn=None)
                egomotion_avg = tf.reduce_mean(egomotion_pred, [1, 2])
                # Tinghui found that scaling by a small constant facilitates training.
                egomotion_final = 0.01 * tf.reshape(
                egomotion_avg, [-1, num_egomotion_vecs, EGOMOTION_VEC_SIZE])

                return egomotion_final