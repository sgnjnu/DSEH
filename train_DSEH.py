#utf-8
import tensorflow._api.v2.compat.v1 as tf
import scipy.io as sio
import numpy as np
import readdatatools as rt
from DataLoader import MyDataLoader
import os
tf.disable_v2_behavior()

feature_dim = 2048
bit_length =16
batch_size = 128
num_epoch = 200
support_size=100
display_step = 20

bit = '%d' % bit_length
HIDDEN_COUNT = 512
Query_dim=128
HIDDEN_COUNT1 = 512
method_name='DSEH'
dataset_name='mir'
pos_weight1=20
n_classes = 24
feature_dim1=1386

alpha=10
beta=0.00001
gama=0.000001
phi=1
lam=1
margin=0.3
dis_margin=0.4
tau=1

input_ix = tf.placeholder(tf.float32, [None, feature_dim])
input_tx = tf.placeholder(tf.float32, [None, feature_dim1])
input_iy = tf.placeholder(tf.float32, [None, feature_dim])
input_ty = tf.placeholder(tf.float32, [None, feature_dim1])
input_iz = tf.placeholder(tf.float32, [None, feature_dim])
input_tz = tf.placeholder(tf.float32, [None, feature_dim1])
input_ixB= tf.placeholder(tf.float32, [None, bit_length])
input_txB= tf.placeholder(tf.float32, [None, bit_length])
input_iyB= tf.placeholder(tf.float32, [None, bit_length])
input_tyB= tf.placeholder(tf.float32, [None, bit_length])
input_izB= tf.placeholder(tf.float32, [None, bit_length])
input_tzB= tf.placeholder(tf.float32, [None, bit_length])

img_component_factor=tf.random_uniform(shape=tf.shape(input_ix),minval=0.0,maxval=2)
txt_component_factor=tf.random_uniform(shape=tf.shape(input_tx),minval=0.0,maxval=2)

o_input_i=input_ix+img_component_factor*input_iy
o_input_t=input_tx+txt_component_factor*input_ty

pre_x = tf.placeholder(tf.float32, [None, n_classes])
pre_y = tf.placeholder(tf.float32, [None, n_classes])
pre_z = tf.placeholder(tf.float32, [None, n_classes])

pos_weight = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

Sxyz_=tf.placeholder(tf.float32, [None,])
Sxyz=tf.reshape(Sxyz_, [-1, 1])
Syxz_=tf.placeholder(tf.float32, [None,])
Syxz=tf.reshape(Syxz_, [-1, 1])
Szxy_=tf.placeholder(tf.float32, [None,])
Szxy=tf.reshape(Szxy_, [-1, 1])

Sxy_=tf.placeholder(tf.float32, [None, ])
Sxy=tf.reshape(Sxy_, [-1, 1])
Sxz_=tf.placeholder(tf.float32, [None, ])
Sxz=tf.reshape(Sxz_, [-1, 1])
Syz_=tf.placeholder(tf.float32, [None, ])
Syz=tf.reshape(Syz_, [-1, 1])


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


def cosine(code1,code2):
    inner= tf.reduce_sum(tf.multiply(code1,code2),reduction_indices=1, keep_dims=True)
    c1_norm = tf.sqrt(tf.reduce_sum(tf.square(code1), reduction_indices=1, keep_dims=True))
    c2_norm = tf.sqrt(tf.reduce_sum(tf.square(code2), reduction_indices=1, keep_dims=True))
    return tf.divide(inner,tf.multiply(c1_norm,c2_norm)+1e-6)


def feature_discriminators(input_feature):
    cls_out = tf.nn.bias_add(tf.matmul(input_feature, f_d_fc1w), f_d_fc1b)
    cls_prob = tf.nn.sigmoid(cls_out)
    return cls_prob

def hash_discriminators(hash_feature):
    cls_out = tf.nn.bias_add(tf.matmul(hash_feature, h_d_fc1w), h_d_fc1b)
    cls_prob = tf.nn.sigmoid(cls_out)
    return cls_prob

def dis_loss(p_pro,n_pro):
    loss=-tf.reduce_mean(tf.log(1e-6+p_pro)+tf.log(1e-6+1-n_pro))
    return loss


def quantization_loss(like_hash,code):
    q_square_loss=tf.reduce_sum(tf.square(like_hash-code),reduction_indices=1, keep_dims=True)
    return tf.reduce_mean(q_square_loss)


def triplet_bound_loss(ci1,ci2,ci3,ct1,ct2,ct3,a123,a213,a312,s12,s23,s13):
    Dxy_it=cosine(ci1,ct2)
    Dxy_ti=cosine(ct1,ci2)

    Dxz_it=cosine(ci1,ct3)
    Dxz_ti=cosine(ct1,ci3)

    Dyz_it=cosine(ci2,ct3)
    Dyz_ti=cosine(ct2,ci3)
    # triplet x,y,z
    loss_xyz_itt=trilpet_sub_loss(Dxy_it,Dxz_it,a123)
    loss_xyz_tii=trilpet_sub_loss(Dxy_ti,Dxz_ti,a123)
    # triplet y,x,z
    loss_yxz_itt=trilpet_sub_loss(Dxy_it,Dyz_it,a213)
    loss_yxz_tii=trilpet_sub_loss(Dxy_ti,Dyz_ti,a213)
    # triplet z,x,y
    loss_zxy_itt=trilpet_sub_loss(Dxz_it,Dyz_it,a312)
    loss_zxy_tii=trilpet_sub_loss(Dxz_ti,Dyz_ti,a312)

    # dis_loss
    loss_xy=dissimilar_loss(Dxy_it,s12)+dissimilar_loss(Dxy_ti,s12)
    loss_yz=dissimilar_loss(Dyz_it,s23)+dissimilar_loss(Dyz_ti,s23)
    loss_xz=dissimilar_loss(Dxz_it,s13)+dissimilar_loss(Dxz_ti,s13)


    return loss_xyz_itt+loss_xyz_tii+loss_yxz_itt+loss_yxz_tii+loss_zxy_itt+loss_zxy_tii+loss_xy+loss_yz+loss_xz



def trilpet_sub_loss(D12,D13,A123):
    P123=(D13-D12)
    S_123=tf.sign(A123)
    loss_sign=tf.abs(S_123)
    trloss=tf.reduce_mean(loss_sign*tf.nn.relu(S_123*P123+margin))
    return trloss


def dissimilar_loss(Dis,Sim):
    dis_sign=tf.cast(tf.equal(Sim,0.0),dtype=tf.float32)
    dis_loss=tf.reduce_mean(dis_sign*tf.nn.relu(0.0-(Dis+dis_margin)))
    return dis_loss

def dissimilar_loss1(x,y):
    dis=cosine(x,y)
    dis_loss=tf.reduce_mean(tf.nn.relu(dis))
    return dis_loss

def rec_loss(x,rec_x):
    return tf.reduce_sum(tf.square(x-rec_x))



ix_trn=transformer_layer(input_ix,i_Qw,i_Kw,i_Vw,i_Rw1,i_Rw2)
iy_trn=transformer_layer(input_iy,i_Qw,i_Kw,i_Vw,i_Rw1,i_Rw2)
iz_trn=transformer_layer(input_iz,i_Qw,i_Kw,i_Vw,i_Rw1,i_Rw2)
io_trn=transformer_layer(o_input_i,i_Qw,i_Kw,i_Vw,i_Rw1,i_Rw2)

tx_trn=transformer_layer(input_tx,t_Qw,t_Kw,t_Vw,t_Rw1,t_Rw2)
ty_trn=transformer_layer(input_ty,t_Qw,t_Kw,t_Vw,t_Rw1,t_Rw2)
tz_trn=transformer_layer(input_tz,t_Qw,t_Kw,t_Vw,t_Rw1,t_Rw2)
to_trn=transformer_layer(o_input_t,t_Qw,t_Kw,t_Vw,t_Rw1,t_Rw2)


totall_rec_loss=rec_loss(input_ix,ix_trn)+rec_loss(input_iy,iy_trn)+rec_loss(input_iz,iz_trn)+rec_loss(input_tx,tx_trn)\
         +rec_loss(input_ty,ty_trn)+rec_loss(input_tz,tz_trn)

ixf,ix_hash = image_hash(input_ix,ix_trn)
iyf,iy_hash = image_hash(input_iy,iy_trn)
izf,iz_hash = image_hash(input_iz,iz_trn)
iof,io_hash = image_hash(o_input_i,io_trn)
txf,tx_hash = text_hash(input_tx,tx_trn)
tyf,ty_hash = text_hash(input_ty,ty_trn)
tzf,tz_hash = text_hash(input_tz,tz_trn)
tof,to_hash = text_hash(o_input_t,to_trn)

ixfpro=feature_discriminators(ixf)
iyfpro=feature_discriminators(iyf)
izfpro=feature_discriminators(izf)
txfpro=feature_discriminators(txf)
tyfpro=feature_discriminators(tyf)
tzfpro=feature_discriminators(tzf)

ixhpro=hash_discriminators(ix_hash)
iyhpro=hash_discriminators(iy_hash)
izhpro=hash_discriminators(iz_hash)
txhpro=hash_discriminators(tx_hash)
tyhpro=hash_discriminators(ty_hash)
tzhpro=hash_discriminators(tz_hash)


Ltr=triplet_bound_loss(ix_hash,iy_hash,iz_hash,tx_hash,ty_hash,tz_hash,Sxyz,Syxz,Szxy,Sxy,Syz,Sxz)
Lqan=quantization_loss(ix_hash,input_ixB)+quantization_loss(iy_hash,input_iyB)+quantization_loss(iz_hash,input_izB)\
   +quantization_loss(tx_hash,input_txB)+quantization_loss(ty_hash,input_tyB)+quantization_loss(tz_hash,input_tzB)
Ladvf = dis_loss(ixfpro,txfpro)+dis_loss(iyfpro,tyfpro)+dis_loss(izfpro,tzfpro)
Ladvh = dis_loss(ixhpro,txhpro)+dis_loss(iyhpro,tyhpro)+dis_loss(izhpro,tzhpro)
Lopen=dissimilar_loss1(input_txB,io_hash)+dissimilar_loss1(input_tyB,io_hash)+dissimilar_loss1(input_tzB,io_hash)+\
      0.001*(dissimilar_loss1(input_ixB,to_hash)+dissimilar_loss1(input_iyB,to_hash)+dissimilar_loss1(input_izB,to_hash))

Lgen=alpha*Ltr+beta*Lqan+tau*Lopen
Ltolfh=Lgen-gama*Ladvf-gama*Ladvh

learning_rate = 0.0001
opt_rec = tf.train.AdamOptimizer(learning_rate*10).minimize(totall_rec_loss,var_list=g_var)
opt_dis = tf.train.AdamOptimizer(learning_rate).minimize(Ladvf+Ladvh,var_list=dis_var_list)
opt_fh = tf.train.AdamOptimizer(learning_rate).minimize(Ltolfh,var_list=fh_var)
my_opts=[opt_dis,opt_fh]

def coding_process(session,data_loader,Wi,Wt,is_first=True):
    code_i = np.zeros([1, bit_length])
    code_t = np.zeros([1, bit_length])
    batch_index = 1
    data_num = data_loader.train_data_num
    perm, locations = rt.spilt_locations_non_perm(data_num, 400)
    while batch_index <= (len(locations) - 1):
        data_indexs = perm[locations[batch_index - 1]:locations[batch_index]]
        batch_is = data_loader.image_traindata[data_indexs, :]
        batch_ts = data_loader.text_traindata[data_indexs, :]
        i_code,t_code= session.run([ix_hash,tx_hash], feed_dict={input_ix: batch_is,input_tx: batch_ts,keep_prob:1.0})
        code_i = np.vstack((code_i, i_code))
        code_t = np.vstack((code_t, t_code))
        batch_index += 1
    code_i = code_i[1:, :]
    code_t = code_t[1:, :]
    if is_first:
        iB = np.sign(code_i)
        tB = np.sign(code_t)
    else:
        iB=np.sign(learning_rate*code_i+np.matmul(data_loader.train_label_list,Wi))
        tB=np.sign(learning_rate*code_t+np.matmul(data_loader.train_label_list,Wt))
    return iB,tB


def getW(B,data_loader,N,lm):
    L=data_loader.train_label_list
    A=np.matmul(np.transpose(L),L)
    noise=np.eye(N)
    nW=np.matmul(np.matmul(np.linalg.inv(A+lm*noise),np.transpose(L)),B)
    return nW

def train_rec(epochs, session,opt,data_loader,trn_step):
    train_batch_numbers = data_loader.train_batch_numbers
    data_loader.shuffle_train_data()
    a_train_step = trn_step
    batch_index = 1
    while batch_index <= train_batch_numbers:
        ix, tx, lx, iy, ty, ly,iz, tz, lz = data_loader.quick_fetch_train_triplets()
        session.run(opt,feed_dict={input_ix:ix,input_iy:iy,input_iz:iz,input_tx:tx,input_ty:ty,input_tz:tz,
                                   pre_x:lx,pre_y:ly,pre_z:lz})

        if batch_index % display_step == 0:
            trn_cost= session.run(totall_rec_loss,feed_dict={input_ix:ix,input_iy:iy,input_iz:iz,input_tx:tx,input_ty:ty,input_tz:tz,
                                   pre_x:lx,pre_y:ly,pre_z:lz})
            rt.print_results_DSEH(batch_index, epochs, trn_cost)
        batch_index += 1
    return a_train_step

def train(epochs, session,opts,data_loader,trn_step,Bi,Bt):

    train_batch_numbers = data_loader.train_batch_numbers
    data_loader.shuffle_train_data()

    a_train_step = trn_step
    batch_index = 1
    while batch_index <= train_batch_numbers:
        ix, tx, lx, iy, ty, ly,iz, tz, lz = data_loader.fetch_train_triplets()
        iBx,iBy,iBz=data_loader.get_xyz_B(Bi)
        tBx,tBy,tBz=data_loader.get_xyz_B(Bt)
        s123,s213,s312,s12,s13,s23=rt.DSEH_triplet_weights(lx,ly,lz)
        for opt in opts:
            session.run(opt,feed_dict={input_ix:ix,input_iy:iy,input_iz:iz,input_tx:tx,input_ty:ty,input_tz:tz,
                                   pre_x:lx,pre_y:ly,pre_z:lz,input_ixB:iBx,input_iyB:iBy,input_izB:iBz,input_txB:tBx,
                                   input_tyB:tBy,input_tzB:tBz,Sxyz_:s123,Syxz_:s213,Szxy_:s312,Sxy_:s12,Sxz_:s13,Syz_:s23
                                       })

        if batch_index % display_step == 0:
            trn_cost,adv_cost,b1,b2= session.run([Lgen,Ladvf,ix_hash,tx_hash],feed_dict={input_ix:ix,input_iy:iy,input_iz:iz,input_tx:tx,input_ty:ty,input_tz:tz,
                                   pre_x:lx,pre_y:ly,pre_z:lz,input_ixB:iBx,input_iyB:iBy,input_izB:iBz,input_txB:tBx,
                                   input_tyB:tBy,input_tzB:tBz,Sxyz_:s123,Syxz_:s213,Szxy_:s312,Sxy_:s12,Sxz_:s13,Syz_:s23
                                   })
            rt.print_results_DSEH(batch_index, epochs, trn_cost)

        batch_index += 1
    if epochs % num_epoch == 0:
        ee = '%d' % epochs
        bit = '%d' % bit_length
        save_path = "./model/" + dataset_name + "/" + bit + "bit/" + method_name + "/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = "./model/"+dataset_name+"/" + bit + "bit/" + method_name+ "/save_" + ee + "_pos.ckpt"
        saver.save(session, save_name)
        print("-------Save Finished!---------")
    return a_train_step


'''prepare data'''

init = tf.initialize_all_variables()
my_data_loader=MyDataLoader(dataset_name,batch_size)

saver = tf.train.Saver()
saver_G=tf.train.Saver([i_Qw,i_Kw,i_Vw,i_Rw1,i_Rw2,t_Qw,t_Kw,t_Vw,t_Rw1,t_Rw2])
G_save_path="./pre_G/" + dataset_name + "/"
if not os.path.exists(G_save_path):
    os.makedirs(G_save_path)
ee1= '%d' % (num_epoch*3)
G_save_name = G_save_path+ "/save_" + ee1 + "_pos.ckpt"

with tf.Session() as sess:
    sess.run(init)
    # train transformer AE rec
    aaa=os.path.isfile(G_save_name)
    aaa1=os.path.exists(G_save_name)
    ckpt = tf.train.get_checkpoint_state(G_save_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver_G.restore(sess, ckpt.model_checkpoint_path)
    else:
        train_step = 1
        epochs=1
        while epochs <= (num_epoch*3):
            a_train_step=train_rec(epochs, sess,opt_rec, my_data_loader,train_step)
            train_step=a_train_step
            epochs+=1
        saver_G.save(sess, G_save_name)

    # train hashing
    train_step = 1
    epochs=1
    iB, tB = coding_process(sess, my_data_loader,0,0,is_first=True)
    while epochs <= num_epoch:

        a_train_step=train(epochs, sess,my_opts, my_data_loader,train_step,iB,tB)
        train_step=a_train_step
        epochs+=1

        if epochs%2==0:
            Win=getW(iB,my_data_loader,n_classes,lam)
            Wtn=getW(tB,my_data_loader,n_classes,lam)
            iB,tB = coding_process(sess, my_data_loader,Win,Wtn,is_first=False)
    code_path = './hash_code/' + dataset_name + '/' + method_name + '/' + str(bit_length) + 'bit/training_phase/'
    if not os.path.exists(code_path):
        os.makedirs(code_path)
    train_feat_path_I = code_path + 'img_trn.mat'
    train_feat_path_T = code_path + 'txt_trn.mat'
    sio.savemat(train_feat_path_I, {'train_feat': iB})
    sio.savemat(train_feat_path_T, {'train_feat': tB})
