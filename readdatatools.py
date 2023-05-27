# coding=utf-8
import numpy as np

def spilt_locations(num_data, batch_size):
    perm = np.random.permutation(num_data)
    if num_data % batch_size != 0:
        perm2 = np.random.permutation(num_data)
        perm2 = perm2 - 1
        lack_num = batch_size - (num_data % batch_size)
        perm = np.append(perm, perm2[0:lack_num])
        print (len(perm))
    locations = []

    for ii in range(0, len(perm) - 1):
        if ii % batch_size == 0:
            locations.append(ii)

    locations.append(len(perm))
    return perm, locations


def spilt_locations_non_perm(num_data, batch_size):
    perm = np.arange(num_data)
    """if num_data % batch_size != 0:
        perm2 = np.random.permutation(num_data)
        perm2 = perm2 - 1
        lack_num = batch_size - (num_data % batch_size)
        perm = np.append(perm, perm2[0:lack_num])
        print len(perm)"""
    locations = []

    for ii in range(0, len(perm) - 1):
        if ii % batch_size == 0:
            locations.append(ii)

    locations.append(len(perm))
    return perm, locations

def print_results_DSEH(batch_index,epochs,trn_cost):
    print("Iter/epochs " + str(batch_index) + "/" + str(
        epochs) + " loss= " + "{:.5f}".format(
        trn_cost))


def DSEH_triplet_weights(labels1, labels2, labels3):
    """Reads tag labele1, labels2, labels3
    Args:
    tags: labels1, labels2, Labels3 share the same dimension [N x L]
    Returns:
    s12,s13,s23= |l1 n L2|,|l1 n L2|, in [0,L]
    s123=sign(s12-s13), s213=sign(s12-s23), s312=sign(s13-s23)
    """
    n1 = np.sum(np.asarray(labels1,dtype=np.float32), axis=1)
    n2 = np.sum(np.asarray(labels1,dtype=np.float32), axis=1)
    n3 = np.sum(np.asarray(labels1,dtype=np.float32), axis=1)
    max_n=np.maximum(np.maximum(n1,n2),n3)
    z=max_n
    n12_ = np.sum(np.asarray(labels1*labels2,
                             dtype=np.float32), axis=1)
    n13_ = np.sum(np.asarray(labels1*labels3,
                             dtype=np.float32), axis=1)
    n23_ = np.sum(np.asarray(labels2*labels3,
                             dtype=np.float32), axis=1)
    sim_123=np.divide(n12_-n13_,z)
    sim_213=np.divide(n12_-n23_,z)
    sim_312=np.divide(n13_-n23_,z)
    s12=n12_
    s13=n13_
    s23=n23_
    return sim_123,sim_213,sim_312,s12,s13,s23

def cosine_sim(c1,c2):
    inner = np.sum(np.multiply(c1, c2), axis=1, keepdims=True)
    c1_norm = np.sqrt(np.sum(np.square(c1), axis=1, keepdims=True))
    c2_norm = np.sqrt(np.sum(np.square(c2), axis=1, keepdims=True))
    return np.divide(inner, np.multiply(c1_norm, c2_norm))
