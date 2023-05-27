#utf-8
import tensorflow._api.v2.compat.v1 as tf
import scipy.io as sio
import numpy as np
from DataLoader import MyDataLoader
import os
tf.disable_v2_behavior()


feature_dim = 2048
bit_length=16
batch_size = 1024
num_epoch = 200
support_size=100
display_step = 20
bit = '%d' % bit_length
HIDDEN_COUNT = 512
Query_dim=128
HIDDEN_COUNT1 = 512
method_name='DSEH'
dataset_name='mir'
data_dataset_name='mir' # the dataset extracted feature
n_classes = 24
feature_dim1=1386




dropout = True


input_ix = tf.placeholder(tf.float32, [None, feature_dim])
input_tx = tf.placeholder(tf.float32, [None, feature_dim1])

img_component_factor=tf.random_uniform(shape=tf.shape(input_ix),minval=0.0,maxval=2.0)
txt_component_factor=tf.random_uniform(shape=tf.shape(input_tx),minval=0.0,maxval=2.0)

input_ic=tf.placeholder(tf.float32, [support_size, feature_dim])
input_tc=tf.placeholder(tf.float32, [support_size, feature_dim1])


# adder for x1 and x2
# 4 MLP for hashing
with tf.name_scope('text_hash_network') as scope:
    t_fc1w = tf.Variable(tf.truncated_normal([feature_dim1, HIDDEN_COUNT],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')
    t_fc1b = tf.Variable(tf.constant(0.0, shape=[HIDDEN_COUNT], dtype=tf.float32),
                         trainable=True, name='x_biases')
    t_fc2w = tf.Variable(tf.truncated_normal([2*HIDDEN_COUNT, HIDDEN_COUNT],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')
    t_fc3w = tf.Variable(tf.truncated_normal([HIDDEN_COUNT, bit_length],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')

with tf.name_scope('text_transformer') as scope:
     t_Qw=tf.Variable(tf.truncated_normal([feature_dim1, Query_dim],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')
     t_Kw=tf.Variable(tf.truncated_normal([feature_dim1, Query_dim],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')
     t_Vw=tf.Variable(tf.truncated_normal([feature_dim1, Query_dim],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')
     t_Rw1=tf.Variable(tf.truncated_normal([Query_dim,HIDDEN_COUNT],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')
     t_Rw2=tf.Variable(tf.truncated_normal([HIDDEN_COUNT,feature_dim1],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')


with tf.name_scope('image_hash_network') as scope:
    i_fc1w = tf.Variable(tf.truncated_normal([feature_dim, HIDDEN_COUNT],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')
    i_fc1b = tf.Variable(tf.constant(0.0, shape=[HIDDEN_COUNT], dtype=tf.float32),
                         trainable=True, name='x_biases')
    i_fc2w = tf.Variable(tf.truncated_normal([2*HIDDEN_COUNT, HIDDEN_COUNT],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')
    i_fc3w = tf.Variable(tf.truncated_normal([HIDDEN_COUNT, bit_length],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')

with tf.name_scope('image_transformer') as scope:
     i_Qw=tf.Variable(tf.truncated_normal([feature_dim, Query_dim],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')
     i_Kw=tf.Variable(tf.truncated_normal([feature_dim, Query_dim],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')
     i_Vw=tf.Variable(tf.truncated_normal([feature_dim, Query_dim],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')
     i_Rw1=tf.Variable(tf.truncated_normal([Query_dim,HIDDEN_COUNT],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')
     i_Rw2=tf.Variable(tf.truncated_normal([HIDDEN_COUNT,feature_dim],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')

with tf.name_scope('feature_discriminator') as scope:
    f_d_fc1w = tf.Variable(tf.truncated_normal([HIDDEN_COUNT, 1],
                                           dtype=tf.float32,
                                           stddev=1e-1), name='weights')
    f_d_fc1b = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                       trainable=True, name='biases')

with tf.name_scope('hash_discriminator') as scope:
    h_d_fc1w = tf.Variable(tf.truncated_normal([bit_length, 1],
                                           dtype=tf.float32,
                                           stddev=1e-1), name='weights')
    h_d_fc1b = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                       trainable=True, name='biases')


g_var=[i_Qw,i_Kw,i_Vw,i_Rw1,i_Rw2,t_Qw,t_Kw,t_Vw,t_Rw1,t_Rw2]
fh_var =[t_fc1w,t_fc1b,t_fc2w,i_fc1w, i_fc1b, i_fc2w,t_fc3w,i_fc3w]
dis_var_list=[f_d_fc1w,f_d_fc1b,h_d_fc1w,h_d_fc1b]



def transformer_layer(feature,Qw,Kw,Vw,R1,R2):
    Q=tf.matmul(feature,Qw)
    K=tf.matmul(feature,Kw)
    V=tf.matmul(feature,Vw)
    weight=tf.nn.softmax(tf.matmul(Q,K,transpose_b=True))
    layer_out1=tf.matmul(weight,V)
    layer_out2=tf.matmul(layer_out1,R1)
    layer_out=tf.matmul(layer_out2,R2)
    return layer_out

def image_hash(x_image,x_exten_f):
    fc1l = tf.nn.bias_add(tf.matmul(x_image, i_fc1w), i_fc1b)
    fc1l_e = tf.nn.bias_add(tf.matmul(x_exten_f, i_fc1w), i_fc1b)
    fc1l_= tf.concat([fc1l,fc1l_e],1)
    fc1 = tf.nn.tanh(fc1l_)
    fc2l = tf.matmul(fc1, i_fc2w)
    fc2 = tf.nn.tanh(fc2l)
    fc3l = tf.matmul(fc2, i_fc3w)
    hash = tf.nn.tanh(fc3l)
    return fc2,hash

def text_hash(x_text,x_exten_f):
    fc1l = tf.nn.bias_add(tf.matmul(x_text, t_fc1w), t_fc1b)
    fc1l_e = tf.nn.bias_add(tf.matmul(x_exten_f, t_fc1w), t_fc1b)
    fc1l_= tf.concat([fc1l,fc1l_e],1)
    fc1 = tf.nn.tanh(fc1l_)
    fc2l = tf.matmul(fc1, t_fc2w)
    fc2 = tf.nn.tanh(fc2l)
    fc3l = tf.matmul(fc2, t_fc3w)
    hash = tf.nn.tanh(fc3l)
    return fc2,hash

ix_trn=transformer_layer(input_ix,i_Qw,i_Kw,i_Vw,i_Rw1,i_Rw2)
tx_trn=transformer_layer(input_tx,t_Qw,t_Kw,t_Vw,t_Rw1,t_Rw2)

ixf,ix_hash = image_hash(input_ix,ix_trn)
txf,tx_hash = text_hash(input_tx,tx_trn)

'''prepare data'''
my_data_loader=MyDataLoader(data_dataset_name,batch_size,is_train=False)
code_path = './hash_code/' + dataset_name + '/' + method_name + '/' + str(
    bit_length) + 'bit/'
if not dataset_name==data_dataset_name:
    code_path=code_path+data_dataset_name+'/'


if not os.path.exists(code_path):
    os.makedirs(code_path)
train_h_path_I = code_path+'img_trn.mat'
test_h_path_I = code_path+'img_tst.mat'


train_h_path_T = code_path+'txt_trn.mat'
test_h_path_T = code_path+'txt_tst.mat'
if not os.path.exists(code_path):
    os.makedirs(code_path)

feature_len=bit_length
train_h_I = np.zeros([1, feature_len])
train_fp_I=np.zeros([1, 1])
train_hp_I=np.zeros([1, 1])
test_h_I = np.zeros([1, feature_len])
test_fp_I=np.zeros([1, 1])
test_hp_I=np.zeros([1, 1])

train_h_T = np.zeros([1, feature_len])
train_fp_T=np.zeros([1, 1])
train_hp_T=np.zeros([1, 1])
test_h_T = np.zeros([1, feature_len])
test_fp_T=np.zeros([1, 1])
test_hp_T=np.zeros([1, 1])

saver = tf.train.Saver()

with tf.Session() as sess:
    save_name = "./model/" + dataset_name + "/" + str(bit_length)+ "bit/" + method_name + "/save_" + str(num_epoch) + "_pos.ckpt"
    saver.restore(sess,save_name)
    tst_batch_index = 1
    batch_index = 1
    while batch_index <= my_data_loader.train_batch_numbers:
        # 获取批数据
        i1,t1,l1 = my_data_loader.fetch_train_data()
        tr_code_i,tr_code_t,i_rec,t_rec = sess.run([ix_hash,tx_hash,ix_trn,tx_trn], feed_dict={input_ix: i1,input_tx: t1})
        train_h_T = np.vstack((train_h_T, tr_code_t))
        train_h_I = np.vstack((train_h_I, tr_code_i))

        print(batch_index)
        batch_index += 1
    train_h_I=train_h_I[1:,:]
    train_fp_I=train_fp_I[1:,:]
    train_hp_I=train_hp_I[1:,:]
    sio.savemat(train_h_path_I, {'train_feat': train_h_I})
    train_h_T = train_h_T[1:, :]
    train_fp_T=train_fp_T[1:,:]
    train_hp_T=train_hp_T[1:,:]
    sio.savemat(train_h_path_T, {'train_feat': train_h_T})


    while tst_batch_index <= my_data_loader.test_batch_numbers:
        i1,t1,l1  = my_data_loader.fetch_test_data()
        ts_code_i,ts_code_t= sess.run([ix_hash,tx_hash], feed_dict={input_ix: i1,input_tx: t1})
        test_h_T = np.vstack((test_h_T, ts_code_t))
        test_h_I = np.vstack((test_h_I, ts_code_i))
        print (tst_batch_index)
        tst_batch_index += 1
    test_h_I=test_h_I[1:,:]
    test_fp_I=test_fp_I[1:,:]
    test_hp_I=test_hp_I[1:,:]
    sio.savemat(test_h_path_I, {'test_feat': test_h_I})
    test_h_T = test_h_T[1:, :]
    test_fp_T=test_fp_T[1:,:]
    test_hp_T=test_hp_T[1:,:]
    sio.savemat(test_h_path_T, {'test_feat': test_h_T})




