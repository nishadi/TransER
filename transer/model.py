import logging
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KDTree
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_knn_neighbors(fitting_dataset, query_dataset, k):
  logging.info('Generating KD Tree neighbor index')
  tree_ = KDTree(fitting_dataset, leaf_size=32, metric='euclidean')
  NN_dist, NN_idx = tree_.query(query_dataset, k=k)
  return NN_dist, NN_idx


def _get_knn_similarities(src_fts, src_labels, tgt_fts, k):
  NN_dist_s, NN_idx_s = get_knn_neighbors(src_fts, src_fts, k + 2)
  NN_dist_t, NN_idx_t = get_knn_neighbors(tgt_fts, src_fts, k + 2)

  dimension = src_fts.shape[1]
  cf_list = list()
  ct_list = list()

  for j in range(len(src_fts)):
    # calculate confidence score
    expected_label = src_labels.iloc[j]
    NN_label_list = src_labels.iloc[NN_idx_s[j, 1:k + 1]]
    cnf_sim = Counter(NN_label_list)[expected_label] / k
    cf_list.append(cnf_sim)

    # Get src and tgt neighbors
    NN_s = src_fts.iloc[NN_idx_s[j, :k], :]
    NN_t = tgt_fts.iloc[NN_idx_t[j, :k], :]

    # Calculate neighborhood centroid distance similarity
    mu_s = np.mean(NN_s, axis=0)
    mu_t = np.mean(NN_t, axis=0)
    cnt_dist = float(np.linalg.norm(mu_s - mu_t))
    cnt_max_dist = np.sqrt(dimension)
    assert cnt_dist <= cnt_max_dist, cnt_max_dist
    cnt_sim = np.exp(-5.0 * float(cnt_dist / cnt_max_dist))
    ct_list.append(cnt_sim)

  return np.array(cf_list), np.array(ct_list)


def classify(src_features, src_labels, tgt_features, tgt_labels, prob=True,
             rstate=0):
  if len(set(src_labels)) <= 1 or Counter(src_labels)[1] < 20 or \
      Counter(src_labels)[0] < 20:
    logging.info(
      '\nClassifier returned p, r, f 0 because src1 {} src0 {}'.format(
        Counter(src_labels)[1], Counter(src_labels)[0]))
    return list(), list(), list()

  prob_list_list = list()
  pred_list_list = list()

  result_dict_list = list()
  for clf_name, clf_cls, gridsearch_space in [
    ('rf', RandomForestClassifier, {'criterion': ['gini', 'entropy']}),
    ('dt', DecisionTreeClassifier, {'criterion': ['gini', 'entropy']}),
    ('svm', SVC, {'C': [0.1, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001],
                  'kernel': ['rbf']}),
    ('lr', LogisticRegression, {'C': [0.1, 1, 10, 100]})

  ]:

    if clf_name == 'svm' and prob:
      clf = clf_cls(random_state=rstate, probability=True)
    else:
      clf = clf_cls(random_state=rstate)

    search = GridSearchCV(clf, gridsearch_space, n_jobs=30, scoring='f1')
    search.fit(src_features, src_labels)
    tgt_pred = search.predict(tgt_features)

    if prob:
      proba = search.predict_proba(tgt_features)
      prob_list_list.append(np.amax(proba, axis=1))
      pred_list_list.append(tgt_pred)
    else:
      logging.info('\tClassifier {}'.format(clf_name))
      logging.info('\t\tbest params : {}'.format(search.best_params_))
      logging.info('\t\tbest score : {}'.format(search.best_score_))

    p = sklearn.metrics.precision_score(tgt_labels, tgt_pred)
    r = sklearn.metrics.recall_score(tgt_labels, tgt_pred)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(tgt_labels,
                                                      tgt_pred).ravel()

    f = 2 * p * r / (p + r)
    f_star = tp / (fn + fp + tp)

    results = {
      'meta': clf_name,
      'precision': round(p * 100, 2),
      'recall': round(r * 100, 2),
      'f': round(f * 100, 2),
      'fstar': round(f_star * 100, 2)
    }
    result_dict_list.append(results)

  prob_list = np.mean(np.array(prob_list_list), axis=0)

  # Prediction is the mode of all classifiers predictions
  # Median is the same as the mode in a binary array
  pred_list = np.round(np.median(np.array(pred_list_list), axis=0))
  return prob_list, pd.Series(pred_list), result_dict_list


def _get_prob_filtered_fts(fts, labels, prob_list, prob_t):
  if len(labels) == 0:
    return [], []

  cand_indices = list()

  for j, prob in [i for i in sorted(enumerate(prob_list), key=lambda x: x[1],
                                    reverse=True)]:
    # for i, prob in enumerate(prob_list):
    if prob >= prob_t:
      cand_indices.append(j)

  # Balancing classes
  tmp_counter = Counter(labels.iloc[cand_indices])
  pos, neg = tmp_counter[1], tmp_counter[0]
  filtered_cand_indices = list()
  current_neg_count = 0
  for i in cand_indices:
    if neg > pos:
      if labels.iloc[i] == 0:
        if current_neg_count >= 3 * pos:
          continue
        current_neg_count += 1
      filtered_cand_indices.append(i)
    else:
      filtered_cand_indices.append(i)

  # logging.info(
  #   '\tfiltered:{} full:{}'.format(len(filtered_cand_indices), len(fts)))
  return fts.iloc[filtered_cand_indices, :], labels.iloc[filtered_cand_indices]


def match_data(src_fts, src_labels, tgt_fts, tgt_labels, k=7, t_c=0.9,
               t_l=0.9, t_p=0.99):
  # Instance selector
  cand_indices = list()
  cf_list, ct_list = _get_knn_similarities(src_fts, src_labels, tgt_fts, k)
  for j, (cnf_sim, cnt_sim) in enumerate(zip(cf_list, ct_list)):
    if cnf_sim >= t_c and cnt_sim >= t_l:
      cand_indices.append(j)
  s_fts_knn_filtered = src_fts.iloc[cand_indices, :]
  s_lbl_knn_filtered = src_labels.iloc[cand_indices]

  # Pseudo label generator
  prob_list, pred_list, _ = classify(
    s_fts_knn_filtered, s_lbl_knn_filtered, tgt_fts, tgt_labels)

  # Target domain classifier
  clf2_tgt_fts, clf2_tgt_lbls = _get_prob_filtered_fts(tgt_fts, pred_list,
                                                       prob_list, t_p)

  _, _, result_dict_list = classify(clf2_tgt_fts, clf2_tgt_lbls, tgt_fts,
                                    tgt_labels, prob=False)

  return result_dict_list

def predict(src_dataset, src_link, tgt_dataset, tgt_link, k=7, t_c=0.9,
               t_l=0.9, t_p=0.99):

  result_list = list()

  # Iterate through different blocking samples
  for i in range(3):
    # Retrieve source dataset
    src_dataset_file = 'data/{}/sampled-{}-{}.pkl'.format(src_dataset,
                                                          src_link, i)
    src_feature_dict = pickle.load(open(src_dataset_file, 'rb'))
    src_fts, src_labels = src_feature_dict['features'], src_feature_dict[
      'labels']

    # Retrieve target dataset
    tgt_dataset_file = 'data/{}/sampled-{}-{}.pkl'.format(tgt_dataset,
                                                          tgt_link, i)
    tgt_feature_dict = pickle.load(open(tgt_dataset_file, 'rb'))
    tgt_fts, tgt_labels = tgt_feature_dict['features'], tgt_feature_dict[
      'labels']

    # Match data with TransER
    tmp_result_list = match_data(src_fts, src_labels, tgt_fts, tgt_labels, k,
                                 t_c, t_l, t_p)
    result_list.extend(tmp_result_list)

  print('Source data set {} {}'.format(src_dataset, src_link))
  print('Target data set {} {}'.format(tgt_dataset, tgt_link))
  print('Linkage quality results')

  p_list = [float(r['precision']) for r in result_list]
  print('\tPrecision: {} +- {}'.format(round(np.mean(p_list), 2),
                                       round(np.std(p_list), 2)))

  r_list = [float(r['recall']) for r in result_list]
  print('\tRecall: {} +- {}'.format(round(np.mean(r_list), 2),
                                    round(np.std(r_list), 2)))

  fstar_list = [float(r['fstar']) for r in result_list]
  print('\tF-star: {} +- {}'.format(round(np.mean(fstar_list), 2),
                                    round(np.std(fstar_list), 2)))

  f1_list = [float(r['f']) for r in result_list]
  print('\tF1: {} +- {}'.format(round(np.mean(f1_list), 2),
                                round(np.std(f1_list), 2)))
