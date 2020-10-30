import tensorflow as tf
import numpy as np
import os
from modelx3 import model
import time
import matplotlib.pyplot as plt
import utils
import glob
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ----------------- Parameters setting -----------------------
sceneFile = 'Lytro/30scenes'  # 30scenes Reflective Occlusions
scenePath = './Datasets/' + sceneFile + '/'
modelPath = "./Model/modelx3"
results_path = 'Results/' + sceneFile + '/'
utils.mkdir(results_path)

s_ori, ang_start = 14, 5
t_ori = s_ori
ang_ori = 7
ang_in = 3
ang_out = 7

FLAG_RGB = 0
tone_coef = 1.0
border_cut = 22
up_scale = int((ang_out - 1) / (ang_in - 1))
num_iter = int(np.ceil(np.log(up_scale) / np.log(3)))
pyramid = 3

lf_files = glob.glob(scenePath + '*.png')
log_batch_path = results_path + '/Log.txt'
with open(log_batch_path, 'w') as f:
        f.write("Dataset is %s.\n" % sceneFile)
# -------------------------------------------------------------------
psnr_bacth = [0 for _ in range(len(lf_files))]
ssim_bacth = [0 for _ in range(len(lf_files))]
time_bacth = [0 for _ in range(len(lf_files))]
for i in range(0, len(lf_files)):
    sceneName = lf_files[i]
    sceneName = sceneName[len(scenePath):-4]
    cur_path = results_path + sceneName
    log_path = cur_path + '/Log.txt'
    utils.mkdir(cur_path + '/images/')

    # -------------- Load LF -----------------
    print("Loading light field %d of %d: %s..." % (i + 1, len(lf_files), sceneName))
    fullLF, inputLF = utils.load4DLF(scenePath + sceneName, s_ori, t_ori, ang_start, ang_ori, ang_in, tone_coef, pyramid)
    [hei, wid, chn, s, t] = fullLF.shape

    out_lf = np.zeros([hei, wid, chn, ang_out, ang_out])

    with open(log_path, 'w') as f:
        f.write("Input (scene name: %s) is a %d X %d light field, extracted start from the %d th view. The output will "
                "be a %d X %d light field.\n" % (sceneName, ang_in, ang_in, ang_start, ang_out, ang_out))


    def slice_reconstruction(size, slice, ang_tar):
        # ---------------- Model -------------------- #
        global slice_y
        with sess.as_default():
            slice_ycbcr = utils.rgb2ycbcr(slice)
            if FLAG_RGB:
                slice_y = slice_ycbcr[:, :, :, 0:1]
                slice_cb = slice_ycbcr[:, :, :, 1:2]
                slice_cr = slice_ycbcr[:, :, :, 2:3]

                slice_y = sess.run(y_out, feed_dict={x: slice_y})
                slice_cb = sess.run(y_out, feed_dict={x: slice_cb})
                slice_cr = sess.run(y_out, feed_dict={x: slice_cr})

                slice_ycbcr = np.concatenate((slice_y, slice_cb, slice_cr), axis=-1)
                slice_ycbcr = tf.convert_to_tensor(slice_ycbcr)
                slice_ycbcr = tf.image.resize_bicubic(slice_ycbcr, [ang_tar, size])
                slice_ycbcr = sess.run(slice_ycbcr)
            else:
                slice_p = slice_ycbcr[:, :, :, 0:1]

                slice_ycbcr = tf.convert_to_tensor(slice_ycbcr)
                slice_ycbcr = tf.image.resize_bicubic(slice_ycbcr, [ang_out, size])
                slice_ycbcr = sess.run(slice_ycbcr)

                slice_y = sess.run(y_out, feed_dict={x: slice_p})

                slice_y = tf.convert_to_tensor(slice_y)
                slice_y = tf.image.resize_bicubic(slice_y, [ang_out, size])
                slice_ycbcr[:, :, :, 0:1] = sess.run(slice_y)
            slice = utils.ycbcr2rgb(slice_ycbcr)
            slice = np.minimum(np.maximum(slice, 0), 1)
        return slice


    # -------------- Column reconstruction -----------------
    start1 = time.clock()
    global ang_cur_in, lf_in
    for s in range(0, ang_in):
        cur_s = s * up_scale
        slice3D = inputLF[:, :, :, s, :]

        for i_iter in range(num_iter):
            if i_iter == 0:
                ang_cur_in = ang_in
                lf_in = slice3D
                ang_cur_out = (ang_in - 1) * 3 + 1

            if i_iter == num_iter - 1:
                ang_cur_out = ang_out
            else:
                ang_cur_out = (ang_cur_in - 1) * 3 + 1

            # -------------- Restore graph ----------------
            tf.reset_default_graph()
            x = tf.placeholder(tf.float32, shape=[None, ang_cur_in, wid, 1])
            y_out = model(3, x)
            g = tf.get_default_graph()
            sess = tf.Session(graph=g)
            with sess.as_default():
                saver = tf.train.Saver(tf.global_variables())
                saver.restore(sess, modelPath)

            lf_in = np.transpose(lf_in, (0, 3, 1, 2))
            lf_in = slice_reconstruction(wid, lf_in, ang_cur_out)
            lf_in = np.transpose(lf_in, (0, 2, 3, 1))
            ang_cur_in = ang_cur_out

        out_lf[:, :, :, cur_s:cur_s + 1, :] = np.expand_dims(lf_in, axis=3)
    elapsed1 = (time.clock() - start1)
    tf.reset_default_graph()
    sess.close()

    # -------------- Row reconstruction -----------------

    start2 = time.clock()
    for t in range(0, ang_out):
        if np.mod(t, up_scale) == 0:
            slice3D = inputLF[:, :, :, :, int(t / up_scale)]
        else:
            slice3D = out_lf[:, :, :, ::up_scale, t]

        for i_iter in range(num_iter):
            if i_iter == 0:
                ang_cur_in = ang_in
                lf_in = slice3D
                ang_cur_out = (ang_in - 1) * 3 + 1

            if i_iter == num_iter - 1:
                ang_cur_out = ang_out
            else:
                ang_cur_out = (ang_cur_in - 1) * 3 + 1

            # -------------- Restore graph ----------------
            tf.reset_default_graph()
            x = tf.placeholder(tf.float32, shape=[None, ang_cur_in, hei, 1])
            y_out = model(3, x)
            g = tf.get_default_graph()
            sess = tf.Session(graph=g)
            with sess.as_default():
                saver = tf.train.Saver(tf.global_variables())
                saver.restore(sess, modelPath)

            lf_in = np.transpose(lf_in, (1, 3, 0, 2))
            lf_in = slice_reconstruction(hei, lf_in, ang_cur_out)
            lf_in = np.transpose(lf_in, (2, 0, 3, 1))
            ang_cur_in = ang_cur_out

        out_lf[:, :, :, :, t:t + 1] = np.expand_dims(lf_in, axis=4)
    elapsed2 = (time.clock() - start1)
    tf.reset_default_graph()
    sess.close()

    with open(log_path, 'a') as f:
        f.write("Reconstruction completed within %.2f seconds (%.3f seconds averaged on each view).\n"
                % (elapsed1 + elapsed2, (elapsed1 + elapsed2) / (ang_out * ang_out)))

    # -------------- Evaluation -----------------
    psnr = np.zeros([ang_out, ang_out])
    ssim = np.zeros([ang_out, ang_out])
    
    for s in range(0, ang_out):
        for t in range(0, ang_out):
            cur_im = out_lf[:, :, :, s, t]

            if np.mod(s, up_scale) != 0 or np.mod(t, up_scale) != 0:
                if ang_out == ang_ori:
                    cur_gt = fullLF[:, :, :, s, t]
                    psnr[s, t], ssim[s, t] = utils.metric(cur_im, cur_gt, border_cut)

            plt.imsave(cur_path + '/images/' + 'out_' + str(s + 1) + '_' + str(t + 1) + '.png',
                       np.uint8(out_lf[:, :, :, s, t] * 255))

    psnr_avg = np.average(psnr) * ang_out * ang_out / (ang_out * ang_out - ang_in * ang_in)
    ssim_avg = np.average(ssim) * ang_out * ang_out / (ang_out * ang_out - ang_in * ang_in)
    psnr_bacth[i] = psnr_avg
    ssim_bacth[i] = ssim_avg
    time_bacth[i] = elapsed1 + elapsed2

    print("PSNR and SSIM on synthetic views are %2.3f and %1.4f." % (psnr_avg, ssim_avg))
    with open(log_path, 'a') as f:
        f.write("PSNR and SSIM on synthetic views are %2.3f and %1.4f.\n" % (psnr_avg, ssim_avg))
    with open(log_batch_path, 'a') as f:
        f.write("%s: %2.3f, %1.4f.\n" % (sceneName, psnr_avg, ssim_avg))

print("PSNR and SSIM on the dataset are %2.3f and %1.4f. Time cosuming: %.2f seconds per light field" % (np.average(psnr_bacth), np.average(ssim_bacth), np.average(time_bacth)))
with open(log_batch_path, 'a') as f:
    f.write("PSNR and SSIM on the dataset are %2.3f and %1.4f. Time cosuming: %.2f seconds per light field\n" % (np.average(psnr_bacth), np.average(ssim_bacth), np.average(time_bacth)))
