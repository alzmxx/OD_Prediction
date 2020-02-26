#-*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import random
import numpy as np
import math
import os
import scipy.sparse as sp
import time
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

tf.app.flags.DEFINE_string("data_dir","/home/xi/Documents/Research/OD_prediction/DL_prediction/data/new_gnn_npy/"," base path")
tf.app.flags.DEFINE_string("out_dir", "/home/xi/Documents/Research/OD_prediction/DL_prediction/data/new_gnn_npy/results/20190725/", "Output directory.")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
#tf.app.flags.DEFINE_integer("train_step", 10000000, "Num to train.")
tf.app.flags.DEFINE_integer("train_step", 1000000, "Num to train.")

def create_new_matrix(A):
    I = np.matrix(np.eye(A.shape[0]))
    A_hat = A + I
    D_hat = np.array(np.sum(A_hat, axis=0))[0]
    D_hat = np.matrix(np.diag(D_hat))
    
    new_A = np.linalg.inv(D_hat) * np.array(A_hat)
    
    return new_A

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

save_file = './saved_networks/model.ckpt'



FLAGS = tf.app.flags.FLAGS
train_path = FLAGS.data_dir+'normalized/train/historical/'
train_images = sorted(os.listdir(train_path + 'link_img_npy/'))
#train_his_labels = sorted(os.listdir(train_path + 'his_od_label_npy/'))
train_labels = sorted(os.listdir(train_path + 'od_label_npy/'))
len_train = len(train_images)

test_path = FLAGS.data_dir+'normalized/test/historical/'
test_images = sorted(os.listdir(test_path + 'link_img_npy/'))
#test_his_labels = sorted(os.listdir(test_path + 'his_od_label_npy/'))
test_labels = sorted(os.listdir(test_path + 'od_label_npy/'))
len_test = len(test_images)

real_od_path_train = '/home/xi/Documents/Research/OD_prediction/DL_prediction/data/new_gnn_npy/train/selected_od_label_npy/'
real_od_path_test = '/home/xi/Documents/Research/OD_prediction/DL_prediction/data/new_gnn_npy//test/od_label_npy/'

adjacency = np.load(FLAGS.data_dir + 'matrix/adjacency.npy')	# 50 * 50
incident = np.load(FLAGS.data_dir + 'matrix/incident.npy')		# 26 * 50
node_adjacency = np.load(FLAGS.data_dir + 'matrix/node_adjacency.npy')	# 26 * 26

new_adjacency = tf.convert_to_tensor(create_new_matrix(adjacency), dtype=tf.float32)    # 50 * 50
new_incident = tf.convert_to_tensor(incident, dtype=tf.float32)                         # 26 * 50
new_node_adjacency = tf.convert_to_tensor(create_new_matrix(node_adjacency), dtype=tf.float32)  # 26 * 26


#print(create_new_matrix(adjacency))
#print(incident)


def main(_):

    with tf.Session() as sess:

        if 1:

            features = tf.placeholder(tf.float32, [50, 8])
            #his_od_lables = tf.placeholder(tf.float32, [26, 25])
            his_od_lables_feed = tf.placeholder(tf.float32, [26, 25])
            his_od_lables = tf.reshape(his_od_lables_feed, [-1, 26, 25, 1])

            labels = tf.placeholder(tf.float32, [26, 25])

            # link graph weights
            weight_1 = tf.Variable(tf.random_normal([8, 100], stddev=0.5))
            weight_2 = tf.Variable(tf.random_normal([100, 50], stddev=0.5))
            weight_3 = tf.Variable(tf.random_normal([50, 25], stddev=0.5))

            # line graph weights
            weight_4 = tf.Variable(tf.random_normal([25, 25], stddev=0.5))

            # node graph weights
            weight_5 = tf.Variable(tf.random_normal([25, 50], stddev=0.5))
            weight_6 = tf.Variable(tf.random_normal([50, 25], stddev=0.5))
            #weight_7 = tf.Variable(tf.random_normal([25, 25], stddev=0.5))

            # weighted summation between two data source
            weight_8 = tf.Variable(tf.random_normal([25, 25], stddev=0.5))
            weight_9 = tf.Variable(tf.random_normal([25, 25], stddev=0.5))

            # convolution weights
            weight_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 50],stddev=0.1))
            weight_conv2 = tf.Variable(tf.truncated_normal([3, 3, 50, 25],stddev=0.1))
            weight_conv3 = tf.Variable(tf.truncated_normal([3, 3, 25, 1],stddev=0.1))

            # non line graph weights
            weight_10 = tf.Variable(tf.random_normal([26, 50], stddev=0.5))
            weight_11 = tf.Variable(tf.random_normal([8, 25], stddev=0.5))

            #tf.summary.histogram('Link Weight', weight_8)
            #tf.summary.histogram('Historical OD Weight', weight_9)

            # line graph biases
            bias_1 = tf.Variable(tf.constant(0.1, shape=[100]))
            bias_2 = tf.Variable(tf.constant(0.1, shape=[50]))
            bias_3 = tf.Variable(tf.constant(0.1, shape=[25]))

            # line graph biases
            bias_4 = tf.Variable(tf.constant(0.1, shape=[25]))

            #historical OD biases
            bias_5 = tf.Variable(tf.constant(0.1, shape=[50]))
            bias_6 = tf.Variable(tf.constant(0.1, shape=[25]))
            #bias_7 = tf.Variable(tf.constant(0.1, shape=[25]))

            # weighted summation biases
            bias_8 = tf.Variable(tf.constant(0.1, shape=[25]))

            # convolution biases
            bias_conv1 = tf.Variable(tf.constant(0.1, shape=[50]))
            bias_conv2 = tf.Variable(tf.constant(0.1, shape=[25]))
            bias_conv3 = tf.Variable(tf.constant(0.1, shape=[1]))

            # non line graph biases
            bias_9 = tf.Variable(tf.constant(0.1, shape=[25]))

            #tf.summary.histogram('Bias', bias_8)

        
            # Line graph neural networks
            output_1 = tf.matmul(new_adjacency, tf.matmul(features, weight_1))
            output_1 = output_1 + bias_1
            output_1 = tf.nn.tanh(output_1)     #50 * 100
            tf.summary.histogram('w_1',weight_1)
            tf.summary.histogram('b_1',bias_1)

            output_2 = tf.matmul(new_adjacency, tf.matmul(output_1, weight_2))
            output_2 = output_2 + bias_2
            output_2 = tf.nn.tanh(output_2)     #50 * 50
            tf.summary.histogram('w_2',weight_2)
            tf.summary.histogram('b_2',bias_2)

            output_3 = tf.matmul(new_adjacency, tf.matmul(output_2, weight_3))
            output_3 = output_3 + bias_3
            output_3 = tf.nn.tanh(output_3)     #50 * 25
            tf.summary.histogram('w_3',weight_3)
            tf.summary.histogram('b_3',bias_3)
        
            output_4 = tf.matmul(new_incident, tf.matmul(output_3, weight_4))
            output_4 = output_4 + bias_4
            #output_4 = tf.nn.tanh(output_4)     #26 * 25
            tf.summary.histogram('w_4',weight_4)
            tf.summary.histogram('b_4',bias_4)

            output_5 = tf.matmul(new_node_adjacency, tf.matmul(output_4, weight_5))
            output_5 = output_5 + bias_5
            #output_5 = tf.nn.tanh(output_5)      #26 * 25
            tf.summary.histogram('w_5',weight_5)
            tf.summary.histogram('b_5',bias_5)

            output_6 = tf.matmul(new_node_adjacency, tf.matmul(output_5, weight_6))
            output_6 = output_6 + bias_6         #26 * 25
            tf.summary.histogram('w_6',weight_6)
            tf.summary.histogram('b_6',bias_6)
            
            out_put = output_6      
            
            '''
            # non line graph neural networks
            out_put = tf.matmul(weight_10, tf.matmul(features, weight_11)) + bias_9

            # Fully connected neural networks
            his_od_out_1 = tf.matmul(his_od_lables, weight_5) + bias_5
            his_od_out_2 = tf.matmul(his_od_out_1, weight_6) + bias_6
            his_od_out_3 = tf.matmul(his_od_out_2, weight_7) + bias_7
			'''
            
            # Convolutional neural networks
            his_od_out_1 = tf.nn.conv2d(his_od_lables, weight_conv1, [1,1,1,1], padding='SAME') + bias_conv1
            his_od_out_2 = tf.nn.conv2d(his_od_out_1, weight_conv2, [1,1,1,1], padding='SAME') + bias_conv2
            his_od_out_3 = tf.nn.conv2d(his_od_out_2, weight_conv3, [1,1,1,1], padding='SAME') + bias_conv3
            his_od_out_3 = tf.reshape(his_od_out_3, [26, 25])
            
            #predicted_y = his_od_out_3
            predicted_y = tf.matmul(out_put, weight_8) + tf.matmul(his_od_out_3, weight_9)
            predicted_y = predicted_y + bias_8
            tf.summary.histogram('w_8',weight_8)
            tf.summary.histogram('w_9',weight_9)
            tf.summary.histogram('b_8',bias_8)

            #tf.summary.image('OD_part', tf.reshape(tf.matmul(his_od_out_3, weight_9), [-1, 26, 25, 1]))
            #tf.summary.image('Line_graph_part', tf.reshape(tf.matmul(out_put, weight_8), [-1, 26, 25, 1]))
            #tf.summary.image('Real_OD', tf.reshape(labels, [-1, 26, 25, 1]))

            predicted_y = tf.nn.relu(predicted_y)


            loss = tf.reduce_mean(tf.abs(predicted_y - labels))
            tf.summary.scalar('Loss', loss)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            #optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
            train = optimizer.minimize(loss)

            merged_summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter('~/logs',graph=sess.graph)
        
            init_op = tf.global_variables_initializer()

        saver = tf.train.Saver()
        #for prediction_step in [2, 3]:
        for prediction_step in [1, 2, 3]:

            sess.run(init_op)
            #saver.restore(sess, save_file)
            
            for step in range(FLAGS.train_step):

                input_index = np.random.randint(len_train)

                # two step prediction limitation
                train_label_item = train_images[input_index]
                min_train_item = int(train_label_item.split('.npy')[0].split('_')[2])

                if min_train_item < (38 - (prediction_step - 2)): 

	                input_data = np.load(train_path + 'link_img_npy/' + train_images[input_index])
	                input_his_od_data = np.load(train_path + 'his_od_label_npy/' + train_images[input_index + prediction_step - 1])
	                #input_his_od_data = [input_his_od_data.reshape(26,25,1)]

	                output_label = np.load(real_od_path_train + train_images[input_index + prediction_step - 1])

	                #_, summary_str = sess.run([train, merged_summary_op], feed_dict={features:input_data, labels:output_label})
	                _, summary_str = sess.run([train, merged_summary_op], feed_dict={features:input_data, his_od_lables_feed:input_his_od_data, labels:output_label})
	                #print(sess.run(predicted_y, feed_dict={features:input_data, labels:output_label}))

                if (step+1) % 10000 == 0:

                    summary_writer.add_summary(summary_str, step)
                    

                    test_error_val = []

                    for test_input_index in range(len_test):

                	    # two step prediction limitation
                        test_label_item = test_images[test_input_index]
                        min_test_item = int(test_label_item.split('.npy')[0].split('_')[2])

                        if min_test_item < (38 - (prediction_step - 2)):

	                        test_input_data = np.load(test_path + 'link_img_npy/' + test_images[test_input_index])
	                        test_input_his_od_data = np.load(test_path + 'his_od_label_npy/' + test_images[test_input_index + prediction_step - 1])
	                        #test_input_his_od_data = [test_input_his_od_data.reshape(26,25,1)]

	                        test_output_label = np.load(real_od_path_test + test_images[test_input_index + prediction_step - 1])
	                    
	                        temp_error_val = sess.run(loss, feed_dict={features:test_input_data, his_od_lables_feed:test_input_his_od_data, labels:test_output_label})
	                        #temp_error_val = sess.run(loss, feed_dict={features:test_input_data, labels:test_output_label})

	                        test_error_val.append(temp_error_val)

                    ave_error_val = np.average(np.array(test_error_val))

                    print("step%d loss: %f" % (step, ave_error_val))
            
            
            for test_input_index in range(len_test):

        	    # two step prediction limitation
                test_label_item = test_images[test_input_index]
                min_test_item = int(test_label_item.split('.npy')[0].split('_')[2])

                #print('Current min item: ', test_label_item)

                if min_test_item < (38 - (prediction_step - 2)):

                    test_input_data = np.load(test_path + 'link_img_npy/' + test_images[test_input_index])
                    test_input_his_od_data = np.load(test_path + 'his_od_label_npy/' + test_images[test_input_index + prediction_step - 1])
                    #test_input_his_od_data = [test_input_his_od_data.reshape(26,25,1)]
                    #test_output_label = np.load(real_od_path_test + test_images[test_input_index + prediction_step - 1])
                    #weight_1_vis,created = sess.run([predicted_y, predicted_y], feed_dict={features:test_input_data, his_od_lables_feed:test_input_his_od_data})
                    created = sess.run(predicted_y, feed_dict={features:test_input_data, his_od_lables_feed:test_input_his_od_data})
                    #sns.heatmap(weight_1_vis, annot=False, fmt='.1f', annot_kws={'size':12}, cmap = 'YlGnBu')
                    #sns.heatmap(weight_1_vis , annot=True, fmt='.1f', annot_kws={'size':12}, cmap = 'YlGnBu')
                    np.save(FLAGS.out_dir + 'step_' + str(prediction_step) + '/' + test_images[test_input_index + prediction_step - 1], created)
                #plt.show()
                #time.sleep(10.5)
                #plt.close()
        #saver.save(sess, save_file)

if __name__ == "__main__":
    tf.app.run()