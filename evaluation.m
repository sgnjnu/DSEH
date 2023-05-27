close all;
clear;
% -- settings start here ---
% top K returned sanmples
top_k = 500;
% set folder
project_dir='./';
dataset_dir='./data/';
dataset_name='mir'; 
target_dataset_name='mir' 
label_dir=strcat(dataset_dir,dataset_name,'/');
feature_folder=strcat(project_dir,'/hash_code/',dataset_name,'/');
bit_length=16;
distance_mode=1; %1 hamming, 4 cosine;
method_name='DSEH';
TextVSImage=1; % t2i task or i2t task

test_label_file = strcat(label_dir,'test_labels.mat');
train_label_file =  strcat(label_dir,'train_labels.mat');
load(train_label_file);
load(test_label_file);  
load(train_label_file);
load(test_label_file);  

code_pare_path=strcat(feature_folder,method_name,'/',num2str(bit_length),'bit/')

if TextVSImage
    feat_test_file = strcat(code_pare_path,'txt_tst.mat');
    feat_test_file1 = strcat(code_pare_path,'img_tst.mat');
    feat_train_file = strcat(code_pare_path,'img_trn.mat');
    feat_train_file1 = strcat(code_pare_path,'txt_trn.mat');
else
    feat_test_file = strcat(code_pare_path,'img_tst.mat');
    feat_test_file1 = strcat(code_pare_path,'txt_tst.mat');
    feat_train_file = strcat(code_pare_path,'txt_trn.mat');
    feat_train_file1 = strcat(code_pare_path,'img_trn.mat');
end

load(feat_test_file);
binary_test = (test_feat>0);

load(feat_train_file);
binary_train = (train_feat>0);

load(train_label_file);
load(test_label_file);
trn_label=double(train_labels);
tst_label =double(test_labels);
              
%NDCG
if TextVSImage
    [map ,acg,ndcg]= precision_ndcg( trn_label, binary_train, tst_label,binary_test, 500, 1);
    disp('text vs image');
else
    [map ,acg,ndcg]= precision_ndcg( trn_label, binary_train, tst_label,binary_test, 500, 1);
    disp('image vs text');
end
fprintf('MAP = %f\n',map);
fprintf('ACG = %f\n',acg);
fprintf('NDCG = %f\n',ndcg);


