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

        for i in range(num_items):
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

    def ItemCFPrediction(self,user_id,item_id,kNN):
        # predict the preference of a user for an item
        pred_score = 0

        train = self.train_ui
        similarity = self.item_similarity

        # find the user's rating history
        # ---Case1: user has rated item
        if train[user_id][item_id] != 0:
            pred_score = train[user_id][item_id]
        else:
            neigh_of_item = self.item_neighbor[item_id]
            neigh = neigh_of_item[0:kNN]
            for neigh_item in neigh:
                pred_score = pred_score + train[user_id,neigh_item] * similarity[item_id,neigh_item]
        return pred_score

    def PredictionGroupPreference(self,group_id,item_id,kNN):
        # Predict the group group_id's preference for item_id
        train_ug = self.train_ug
        train_gi = self.train_gi

        # find users of each group
        users = train_ug[:,group_id]
        users_ingroup = np.nonzero(users)
        users_ingroup_idx = users_ingroup[0]
        individual_score = 0     # predict each user's preference for item_Id and aggregate these preference score

        for user in users_ingroup_idx:
            individual_score = individual_score + self.ItemCFPrediction(user,item_id,kNN)
        group_score = individual_score / users_ingroup_idx.shape[0]
        return group_score

    def EvaluateForGroupRec(self,kNN,top_N):
        precision = 0
        recall = 0
        group_count = 0

        test_gi = self.test_gi
        train_gi = self.train_gi
        num_groups = self.num_groups

        for group_id in range(num_groups):
            r_i = test_gi[group_id]                        # firstly identify items that group i has rated
            test_items = np.nonzero(r_i)
            test_items_idx = test_items[0]
            if len(test_items_idx) == 0:            # if this group did not select any items in the test set, then skip the evaluate procedure
                continue
            else:
                r_i_train = train_gi[group_id]
                train_items = np.nonzero(r_i_train)
                train_items_idx = train_items[0]
                pred_item_idx = np.setdiff1d(np.arange(self.num_items),train_items_idx)     # for each group, items that have not been rated in the train set should be predicted
                pred_score = np.zeros(self.num_items)
                # print group_id,pred_item_idx,pred_score
                for item in pred_item_idx:
                    pred_score[item] = self.PredictionGroupPreference(group_id,item,kNN)
                rec_cand = np.argsort(-pred_score)      # candidate set of the recommended items for group_id
                rec_list = rec_cand[0:top_N]
                hit_set = np.intersect1d(rec_list,test_items_idx)
                precision = precision + len(hit_set) / (top_N * 1.0)
                recall = recall + len(hit_set) / (len(test_items_idx) * 1.0)
                group_count = group_count + 1
        precision = precision / (group_count * 1.0)
        recall = recall / (group_count * 1.0)

        return precision,recall

def example_test():
    train_ui = load_matrix('train_ui_sample.txt',6,4)
    train_ug = load_matrix('train_ug_sample.txt',6,3)
    train_gi = load_matrix('train_gi_sample.txt',3,4)
    test_gi = load_matrix('test_gi_sample.txt',3,4)
    kNNItemCF = ItemCF(train_ui,train_ug,train_gi,test_gi)
    kNNItemCF.ItemSimilarity()
    precision,recall = kNNItemCF.EvaluateForGroupRec(2,2)
    print precision,recall

if __name__=='__main__':
    example_test()
