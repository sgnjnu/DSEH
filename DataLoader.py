import numpy as np
import scipy.io as sio
import readdatatools as rt
import os.path


class MyDataLoader( ):
    def __init__(self, dataset_name,batch_size,is_train=True):
        self.dataset_name=dataset_name
        data_project_name='./data/'+dataset_name
        self.is_train=is_train

        image_train_dataFile = data_project_name+'/mir_feat-train.mat'
        image_test_dataFile = data_project_name+'/mir_feat-test.mat'
        text_train_dataFile = data_project_name+'/train_text_data.mat'
        text_test_dataFile = data_project_name+'/test_text_data.mat'
        train_labelFile = data_project_name+'/train_labels.mat'
        test_labelFile = data_project_name+'/test_labels.mat'

        image_traindata = sio.loadmat(image_train_dataFile)
        image_testdata = sio.loadmat(image_test_dataFile)
        train_label_list = sio.loadmat(train_labelFile)
        test_label_list = sio.loadmat(test_labelFile)

        image_traindata = image_traindata["feat_train"]
        image_traindata = np.asarray(image_traindata, dtype=np.float32)
        image_testdata = image_testdata["feat_test"]
        image_testdata = np.asarray(image_testdata, dtype=np.float32)
        train_label_list = train_label_list["train_labels"]
        train_label_list = np.asarray(train_label_list, dtype=np.int32)
        test_label_list = test_label_list["test_labels"]
        test_label_list = np.asarray(test_label_list, dtype=np.int32)
        text_traindata = sio.loadmat(text_train_dataFile)
        text_testdata = sio.loadmat(text_test_dataFile)
        text_traindata = text_traindata["train_text_data"]
        text_traindata = np.asarray(text_traindata, dtype=np.float32)
        text_testdata = text_testdata["test_text_data"]
        text_testdata = np.asarray(text_testdata, dtype=np.float32)


        self.batch_size=batch_size
        self.train_data_num = len(train_label_list)
        if is_train:
            self.image_traindata=image_traindata
            self.image_testdata=image_testdata
            self.text_traindata=text_traindata
            self.text_testdata=text_testdata
            self.train_label_list= train_label_list
            self.test_label_list= test_label_list
            self.train_data_num = len(self.train_label_list)
            self.test_data_num = len(self.test_label_list)
            self.perm1, self.locations1 = rt.spilt_locations(self.train_data_num, self.batch_size)
            self.perm2, self.locations2 = rt.spilt_locations(self.train_data_num, self.batch_size)
            self.perm3, self.locations3 = rt.spilt_locations(self.train_data_num, self.batch_size)
            self.test_perm, self.test_locations = rt.spilt_locations(self.test_data_num, self.batch_size)
            self.train_batch_numbers = len(self.locations1) - 1
            self.test_batch_numbers = len(self.test_locations) - 1
            self.train_batch_index = 1
            self.test_batch_index = 1
        else:
            self.image_traindata=image_traindata
            self.image_testdata=image_testdata
            self.text_traindata=text_traindata
            self.text_testdata=text_testdata
            self.train_label_list= train_label_list
            self.test_label_list= test_label_list
            self.train_data_num = len(self.train_label_list)
            self.test_data_num = len(self.test_label_list)
            self.perm, self.locations = rt.spilt_locations_non_perm(self.train_data_num, self.batch_size)
            self.test_perm, self.test_locations = rt.spilt_locations_non_perm(self.test_data_num, self.batch_size)
            self.train_batch_numbers = len(self.locations) - 1
            self.test_batch_numbers = len(self.test_locations) - 1
            self.train_batch_index = 1
            self.test_batch_index = 1

    def shuffle_train_data(self):
            self.perm1, self.locations1 = rt.spilt_locations(self.train_data_num, self.batch_size)
            self.perm2, self.locations2 = rt.spilt_locations(self.train_data_num, self.batch_size)
            self.perm3, self.locations3 = rt.spilt_locations(self.train_data_num, self.batch_size)
            self.train_batch_index = 1

    def fetch_train_data(self):
        if not self.is_train:
            data_index = self.perm[self.locations[self.train_batch_index - 1]:self.locations[self.train_batch_index]]
        else:
            data_index = self.perm1[self.locations1[self.train_batch_index - 1]:self.locations1[self.train_batch_index]]
        self.train_batch_index=self.train_batch_index+1
        if self.train_batch_index > self.train_batch_numbers:
            self.train_batch_index=1
        image_data= self.image_traindata[data_index , :]
        text_data= self.text_traindata[data_index , :]
        label_list=self.train_label_list[data_index , :]

        return image_data,text_data,label_list

    def fetch_test_data(self):
        data_index = self.test_perm[self.test_locations[self.test_batch_index - 1]:self.test_locations[self.test_batch_index]]
        self.test_batch_index=self.test_batch_index+1
        if self.test_batch_index > self.test_batch_numbers:
            self.test_batch_index=1
        image_data= self.image_testdata[data_index , :]
        text_data= self.text_testdata[data_index , :]
        label_list=self.test_label_list[data_index , :]

        return image_data,text_data,label_list

    def fetch_train_triplets(self):
        a = np.arange(self.train_data_num)
        self.data_index1 = self.perm1[self.locations1[self.train_batch_index - 1]:self.locations1[
            self.train_batch_index]]
        self.train_batch_index = self.train_batch_index + 1
        if self.train_batch_index > self.train_batch_numbers:
            self.train_batch_index = 1
        image_data1 = self.image_traindata[self.data_index1, :]
        text_data1 = self.text_traindata[self.data_index1, :]
        label_list1 = self.train_label_list[self.data_index1, :]
        cur_W = np.matmul(label_list1, self.train_label_list.T)
        cur_sim = np.minimum(cur_W, 1.0)
        sim_column_index = np.tile(a, (1, 1))
        self.data_index2=[]
        self.data_index3=[]
        for ii in range(len(cur_sim)):
            aa=sim_column_index[0, np.where(cur_sim[ii,:] == 1)]
            self.data_index2.append(np.random.choice(np.squeeze(aa)))
            if ii <= 0.99*len(cur_sim):
                self.data_index3.append(np.random.choice(np.squeeze(aa)))
            else:
                bb = sim_column_index[0, np.where(cur_sim[ii, :] == 0)]
                self.data_index3.append(np.random.choice(np.squeeze(bb)))
        image_data2= self.image_traindata[self.data_index2 , :]
        text_data2= self.text_traindata[self.data_index2 , :]
        label_list2=self.train_label_list[self.data_index2 , :]
        image_data3= self.image_traindata[self.data_index3 , :]
        text_data3= self.text_traindata[self.data_index3 , :]
        label_list3=self.train_label_list[self.data_index3 , :]
        return image_data1,text_data1,label_list1,image_data2,text_data2,label_list2,image_data3,text_data3,label_list3

    def quick_fetch_train_triplets(self):
        data_index1 = self.perm1[self.locations1[self.train_batch_index - 1]:self.locations1[self.train_batch_index]]
        data_index2 = self.perm2[self.locations2[self.train_batch_index - 1]:self.locations2[self.train_batch_index]]
        data_index3 = self.perm3[self.locations3[self.train_batch_index - 1]:self.locations3[self.train_batch_index]]
        self.train_batch_index=self.train_batch_index+1
        if self.train_batch_index > self.train_batch_numbers:
            self.train_batch_index=1
        image_data1= self.image_traindata[data_index1 , :]
        text_data1= self.text_traindata[data_index1 , :]
        label_list1=self.train_label_list[data_index1 , :]
        image_data2= self.image_traindata[data_index2 , :]
        text_data2= self.text_traindata[data_index2 , :]
        label_list2 = self.train_label_list[data_index2, :]
        image_data3= self.image_traindata[data_index3 , :]
        text_data3= self.text_traindata[data_index3 , :]
        label_list3 = self.train_label_list[data_index3, :]
        self.data_index1= data_index1
        self.data_index2 = data_index2
        self.data_index3 = data_index3

        return image_data1,text_data1,label_list1,image_data2,text_data2,label_list2,image_data3,text_data3,label_list3



    def get_xyz_B(self,B):
        return B[self.data_index1,:],B[self.data_index2,:],B[self.data_index3,:]


