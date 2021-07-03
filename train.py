import tensorflow as tf
import argparse
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from modelx3 import model
from model_ae import encoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import utils
    

# ----------------Parameters setting-------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--up_scale', type=int, default=3, help='upsampling scale of the network: 3 for modelx3 and 4 for modelx4')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--training_iterations', type=int, default=600000, help='number of training iterations')
parser.add_argument('--train_data_path', type=str, default='./TrainData/train.h5', help='path of the training data')
parser.add_argument('--test_data_path', type=str, default='./TrainData/test.h5', help='path of the testing data')
parser.add_argument('--model_name', type=str, default='modelx3', help='name of the trained model')
parser.add_argument('--model_name_ae', type=str, default='model_ae', help='name of the autoencoder model')
parser.add_argument('--batch_size', type=int, default=28, help='size of the batches for training')
parser.add_argument('--batch_size_test', type=int, default=128, help='size of the batches for testing')
parser.add_argument('--display_interval', type=int, default=100, help='number of iterations for log display and record')
parser.add_argument('--test_interval', type=int, default=1000, help='number of iterations for testing')
parser.add_argument('--pyramid', type=int, default=4, help='down/up-sampling scale in the Fusion net. Fixed!')
opt = parser.parse_args()
print(opt)

model_load_path = "./Model/" + opt.model_name
model_save_path = "./Model/" + opt.model_name
log_path = "./Model/Log_" + opt.model_name + ".txt"
model_ae_load_path = './Model/' + opt.model_name_ae

# ---------------- Load training data -------------------- #
train_label, _ = utils.load_h5_data(opt.train_data_path)
test_label, _ = utils.load_h5_data(opt.test_data_path)

train_num, sizeA, sizeW, _ = train_label.shape
sizeA_in = (sizeA - 1) // opt.up_scale + 1
sizeA = (sizeA_in - 1) * opt.up_scale + 1
sizeW = sizeW // opt.pyramid * opt.pyramid

train_label = train_label[:, :sizeA, :sizeW, :]
train_data = train_label[:, ::opt.up_scale, :, :]
batch_num = int(train_num / opt.batch_size)

test_label = test_label[:, :sizeA, :sizeW, :]
test_data = test_label[:, ::opt.up_scale, :, :]
batch_num_test = int(test_data.shape[0] / opt.batch_size_test)

info = "Training data number is %d, and batch size is %d; testing data number is %d, and batch size is %d." \
       % (batch_num * opt.batch_size, opt.batch_size, batch_num_test * opt.batch_size_test, opt.batch_size_test)
print(info)

with open(log_path, 'w') as f:
    f.write(info + '\n')

# ---------------- Setting networks -------------------- #
x = tf.placeholder(tf.float32, shape=[None, sizeA_in, sizeW, 1])
y_ = tf.placeholder(tf.float32, shape=[None, sizeA, sizeW, 1])

y_out = model(opt.up_scale, x)
feature1, feature2, feature3, feature4 = encoder(y_out, y_)

# ---------------- Criteria -------------------- #
mse = tf.reduce_mean(tf.square(y_ - y_out))
loss_mae = tf.losses.absolute_difference(y_, y_out)
psnr = 20 * tf.reduce_sum(tf.log(1 / np.array(tf.sqrt(mse))) / 2.3026)

loss1 = 0.2 * tf.reduce_mean(feature1)
loss2 = 0.2 * tf.reduce_mean(feature2)
loss3 = 0.1 * tf.reduce_mean(feature3)
loss4 = 0.006 * tf.reduce_mean(feature4)
loss_style = loss1 + loss2 + loss3 + loss4

loss = loss_mae + loss_style

# ---------------- Setting optimizer -------------------- #
num_params = utils.get_num_params('SR') + utils.get_num_params('Blender')
train_vars = tf.trainable_variables()
vars_Reconstructor = [var for var in train_vars if var.name.startswith('SR')]
vars_Blender = [var for var in train_vars if var.name.startswith('Blender')]
SR_solver = tf.train.AdamOptimizer(opt.lr).minimize(loss, var_list=vars_Reconstructor)
Blender_solver = tf.train.AdamOptimizer(opt.lr).minimize(loss, var_list=vars_Blender)
train_step = tf.group(SR_solver, Blender_solver)


# ---------------- Define training -------------------#
def train(my_session, train_data, train_label, model_save_path, iterations):
    minLoss = 100
    epoch = 0
    for i in range(iterations):
        batch_x = train_data[(i % batch_num) * opt.batch_size: (i % batch_num + 1) * opt.batch_size]
        batch_y = train_label[(i % batch_num) * opt.batch_size: (i % batch_num + 1) * opt.batch_size]

        if i % batch_num == 0:
            epoch += 1
        if i % opt.display_interval == 0:
            curLoss, curPSNR = my_session.run([loss, psnr], feed_dict={x: batch_x, y_: batch_y})
            info = "Epoch {}, interation {}, Minibatch loss = {:.5f}, PSNR = {:.2f}.".format(epoch, i, curLoss, curPSNR)
            if i % opt.test_interval == 0:
                test_loss, test_PSNR = test(my_session, test_data, test_label)
                info += " Test loss = {:.5f}, PSNR = {:.2f}".format(test_loss, test_PSNR)

                # Save model
                if test_loss < minLoss:
                    minLoss = test_loss
                    saver.save(my_session, model_save_path)
            print(info)
            with open(log_path, 'a') as f:
                f.write(info + '\n')

        my_session.run(train_step, feed_dict={x: batch_x, y_: batch_y})
    print("Optimization Finished!")
    return my_session


def test(my_session, test_data, test_label):
    test_loss = 0
    test_PSNR = 0
    for j in range(batch_num_test):
        batch_x_test = test_data[(j % batch_num_test) * opt.batch_size_test:
                                 (j % batch_num_test + 1) * opt.batch_size_test]
        batch_y_test = test_label[(j % batch_num_test) * opt.batch_size_test:
                                  (j % batch_num_test + 1) * opt.batch_size_test]
        curTestLoss, curTestPSNR = my_session.run([loss, psnr], feed_dict={x: batch_x_test, y_: batch_y_test})
        test_loss += curTestLoss
        test_PSNR += curTestPSNR
    test_loss = test_loss / batch_num_test
    test_PSNR = test_PSNR / batch_num_test
    return test_loss, test_PSNR


########################################
# ------------ Training -------------- #
########################################
with tf.Session() as sess:
    # ----------- Restore model weights ------------ #
    vars_shear = []
    vars_ae = []
    train_vars = tf.trainable_variables()
    variables_names = [variables.name for variables in train_vars]
    n = 0
    for k in zip(variables_names):
        if 'Blender' in str(k) or 'SR' in str(k):
            vars_shear.append(train_vars[n])
        if 'encoder' in str(k):
            vars_ae.append(train_vars[n])
        n += 1
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    try:
        saver_ae = tf.train.Saver(var_list=vars_ae)
        saver_ae.restore(sess, model_ae_load_path)
        saver_shear = tf.train.Saver(var_list=vars_shear)
        saver_shear.restore(sess, model_load_path)
    except:
        info = 'Restoring previous trained weights failed, starting a new training opt. We have %d parameters.' \
               % num_params
    else:
        info = 'Restoring previous trained weights, continue the training opt. We have %d parameters.' % num_params
    print(info)
    with open(log_path, 'a') as f:
        f.write(info + '\n')

    sess = train(sess, train_data, train_label, model_save_path, opt.training_iterations)
    sess.close()
