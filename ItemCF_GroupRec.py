# coding = utf-8
# Please feel free to contact with me if you have any question with the code.
__author__ = 'wangjinkun@mail.hfut.edu.cn'

import time
import numpy as np
# import scipy.sparse as sparse

def load_matrix(filename, num_users, num_items):
    t0 = time.time()
    matrix = np.zeros((num_users,num_items))
    for line in open(filename):
        user,item = line.split()
        user = int(user)
        item = int(item)
        count = 1.0
        matrix[user-1,item-1] = count
    t1 = time.time()
    # matrix = sparse.csr_matrix(matrix)
    print 'Finished loading matrix in %f seconds' % (t1-t0)
    return  matrix

class ItemCF:

    def __init__(self,train_ui,train_ug,train_gi,test_gi):
        self.train_ui = train_ui
        self.train_ug = train_ug
        self.train_gi = train_gi
        self.test_gi = test_gi
        self.num_users = train_ui.shape[0]
        self.num_items = train_ui.shape[1]
        self.num_groups = train_gi.shape[0]

    def ItemSimilarity(self):
        t0 = time.time()
        train = self.train_ui
        num_items = self.num_items
        self.item_similarity = np.zeros((num_items,num_items))

        for i in np.arange(0,num_items):
            r_i = train[:,i]
            self.item_similarity[i,i] = 0
            for j in np.arange(i+1,num_items):
                r_j = train[:,j]
                num = np.dot(r_i.T , r_j)
                denom = np.linalg.norm(r_i) * np.linalg.norm(r_j)
                if denom == 0:
                    cos = 0
                else:
                    cos = num / denom
                self.item_similarity[i,j] = cos
                self.item_similarity[j,i] = cos
        self.item_neighbor = np.argsort(-self.item_similarity)
        t1 = time.time()
        print 'Finished calculating similarity matrix in %f seconds' % (t1-t0)

    def Recommendation(self,user_id,kNN):
        # recommend a top_N recommendation list for user_id
        train_ui = self.train_ui
        similarity = self.item_similarity

        r_u = train_ui[user_id]
        pred_score = r_u

        rated_items = np.nonzero(r_u)
        rated_items_idx = rated_items[0]    # items rated by user_id in train set
        predict_items_idx = np.setdiff1d(np.arange(0,self.num_items),rated_items_idx)   # item index that has to be predicted
        for i in predict_items_idx:
            item_idx = i
            neighbor_ordered = self.item_neighbor[item_idx] #
            for neigh in neighbor_ordered[0:kNN]:
                pred_score[i] = pred_score[i] + train_ui[user_id,neigh] * similarity[i,neigh]
        return pred_score

    def RecommendationForGroup(self,group_id,kNN,top_N):
        # recommend a top_N recommendation list for group_id
        train_ug = self.train_ug
        train_gi = self.train_gi
        group_id = 1

        # find users of each group
        users = train_ug[:,group_id]
        users_ingroup = np.nonzero(users)
        users_ingroup_idx = users_ingroup[0]
        individual_score = np.zeros((1,self.num_items))
        for user in users_ingroup_idx:
            individual_score = individual_score + self.Recommendation(user,kNN)
        group_score = individual_score / users_ingroup_idx.shape[0]

        # items that the group has given
        r_g = train_gi[group_id]
        rated_items = np.nonzero(r_g)
        rated_items_idx = rated_items[0]
        group_score[0,rated_items_idx] = 0
        rec_candidate = np.argsort(-group_score)
        rec_candidate_X = rec_candidate[0]
        rec_list = rec_candidate_X[0:top_N]
        return rec_list

    def EvaluateForGroupRec(self,kNN,top_N):
        train_ug = self.train_ug
        test_gi = self.test_gi
        num_groups = self.num_groups

        precision = 0
        recall = 0
        group_count = 0

        for i in np.arange(0,1):
            r_i = test_gi[i]                # firstly identify items that group i has rated
            test_items = np.nonzero(r_i)
            test_items_idx = test_items[0]
            if len(test_items_idx) == 0:    # if this group did not select any items in the test set, then skip the evaluate procedure
                continue
            else:
                rec_of_i = self.RecommendationForGroup(i,kNN,top_N)
                hit_set = np.intersect1d(rec_of_i,test_items_idx)
                precision = precision + len(hit_set) / (top_N * 1.0)
                recall = recall + len(hit_set) / (len(test_items_idx) * 1.0)
                group_count = group_count + 1
        precision = precision / (group_count * 1.0)
        recall = recall / (group_count * 1.0)
        return precision,recall

def test():
    train_ui = load_matrix('train_ui_sample.txt',6,4)
    train_ug = load_matrix('train_ug_sample.txt',6,3)
    train_gi = load_matrix('train_gi_sample.txt',3,4)
    test_gi = load_matrix('test_gi_sample.txt',3,4)
    kNNItemCF = ItemCF(train_ui,train_ug,train_gi,test_gi)
    kNNItemCF.ItemSimilarity()
    precision,recall= kNNItemCF.EvaluateForGroupRec(2,2)
    print precision,recall

if __name__=='__main__':
    test()
