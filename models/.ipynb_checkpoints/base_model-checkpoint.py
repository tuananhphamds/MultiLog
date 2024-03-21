import os
import sys
import json
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

HOME = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(HOME + '/preprocess')

from dataloader import DataLoader
from model_utils import ShowLRate
from subsequencegenerator import SubsequenceGenerator


class BaseModel:
    def __init__(self,
                 cfg,
                 embedding_matrix,
                 log_template_dict,
                 hyper_center=None):
        self.cfg = cfg
        self.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.embedding_matrix = embedding_matrix
        self.log_template_dict = log_template_dict
        self.hyper_center = hyper_center
        self.model = None

    def _build_model(self):
        pass

    def _compile(self, num_X_train):
        pass

    def _load_model(self, model_file):
        pass

    def train(self, X_train, Y_train, X_val=None, Y_val=None):
        self._build_model()
        model_file = self.cfg['model_file']

        if model_file is not None:
            model_path = os.path.join(self.cfg['result_path'], model_file)
        else:
            model_path = os.path.join(self.cfg['result_path'], 'best_model.hdf5')

        model_checkpoint = ModelCheckpoint(model_path,
                                           monitor='loss',
                                           verbose=self.cfg['verbose'],
                                           save_best_only=True,
                                           save_weights_only=False,
                                           model='min')

        dataloader = DataLoader(X=X_train,
                                Y=Y_train,
                                batch_size=self.cfg['batch_size'],
                                vocab_size=self.cfg['vocab_size'],
                                log_template_dict=self.log_template_dict,
                                shuffle=True,
                                window_size=self.cfg['window_size'],
                                hyper_center=self.hyper_center,
                                mask_rate=self.cfg['mask_rate'],
                                model_name=self.cfg['model_name'],
                                reduction_dim=self.cfg['reduction_dim'])

        # val_dataloader = DataLoader(X=X_val,
        #                             Y=Y_val,
        #                             batch_size=self.cfg['batch_size'],
        #                             vocab_size=self.cfg['vocab_size'],
        #                             log_template_dict=self.log_template_dict,
        #                             shuffle=True,
        #                             window_size=self.cfg['window_size'],
        #                             hyper_center=self.hyper_center,
        #                             predict_next_only=self.cfg['predict_next_only'],
        #                             contrastive_learning=self.cfg['contrastive_learning'])
        self._compile(len(X_train))
        history = self.model.fit(dataloader,
                                 epochs=self.cfg['epochs'],
                                 callbacks=[model_checkpoint, ShowLRate()],
                                 verbose=self.cfg['verbose'])
        self._save_train_results_as_plot(history.history)

    def _save_normal_abnormal_scores_plot(self, normal_scores, abnormal_scores, title=None):
        print('--------------------SAVING NORMAL ABNORMAL SCORES {} PLOT--------------------'.format(title))
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].hist(normal_scores, bins=100, color='blue', label='normal')
        axes[0].set_title(title)
        axes[0].legend()
        axes[1].hist(abnormal_scores, bins=100, color='red', label='abnormal')
        axes[1].set_title(title)
        axes[1].legend()
        plt.savefig(os.path.join(self.cfg['result_path'], '{}.pdf'.format(title)))

    def _save_train_results_as_plot(self, history):
        print('--------------------SAVING TRAINING LOSSES TO PLOTS--------------------')
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))

        max_min = {0: {'max': [], 'min': []},
                   1: {'max': 1, 'min': 0}}

        for key in history.keys():
            if 'loss' in key:
                max_min[0]['max'].append(max(history[key]))
                max_min[0]['min'].append(min(history[key]))
        max_min[0]['max'] = max(max_min[0]['max'])
        max_min[0]['min'] = min(max_min[0]['min'])

        num_epochs = self.cfg['epochs']
        for key in history.keys():
            if 'loss' in key and 'val' not in key:
                key_ax = axes[0, 0]
                key_ax.set_title('Train losses')
                key_ax.set_ylim(bottom=max_min[0]['min'], top=max_min[0]['max'])
            elif 'loss' in key and 'val' in key:
                key_ax = axes[0, 1]
                key_ax.set_title('Val losses')
                key_ax.set_ylim(bottom=max_min[0]['min'], top=max_min[0]['max'])
            elif 'acc' in key:
                key_ax = axes[1, 0]
                key_ax.set_title('Accuracy')
                key_ax.set_ylim(bottom=max_min[1]['min'], top=max_min[1]['max'])
            else:
                key_ax = axes[1, 1]
                key_ax.set_title('Remainings')
            key_ax.plot(range(num_epochs), history[key], label=key)
            key_ax.text(0, history[key][0], str(round(history[key][0], 4)))
            key_ax.text(num_epochs - 1, history[key][-1], str(round(history[key][-1], 4)))
            key_ax.legend()

        plt.savefig(os.path.join(self.cfg['result_path'], 'train_plots.pdf'))

    def _save_best_results(self, best_results):
        print('--------------------SAVING BEST RESULTS--------------------')
        with open(os.path.join(self.cfg['result_path'], 'best_results.pkl'), 'wb') as f:
            pickle.dump(best_results, f)

    def test(self, normal_sess_dct, abnormal_sess_dct, X_val=None, Y_val=None):
        pass

    def _calculate_scores_per_session(self, sess_dct, generator_batch_size, all_scores, top_k=1, min=True, mask=True):
        sub_generator = SubsequenceGenerator(sess_dct,
                                             batch_size=generator_batch_size)

        if min:
            agg_func = np.min
        else:
            agg_func = np.max

        total_scores = []
        freq_per_sess = []

        for idx, batch_data in enumerate(sub_generator.get_batches()):
            batch_probs = all_scores[idx]

            if mask:
                batch_probs = batch_probs[:, :top_k]
                batch_probs = np.mean(batch_probs, axis=-1)

            for i in range(len(batch_data)):
                start_position = batch_data[i][2]
                end_position = batch_data[i][3]
                seq = batch_data[i][4]
                if start_position == end_position:
                    total_scores.append(agg_func(batch_probs[start_position]))
                else:
                    total_scores.append(agg_func(batch_probs[start_position:end_position]))

                freq_per_sess.append(sess_dct[seq])

        assert len(total_scores) == len(freq_per_sess)
        return total_scores, freq_per_sess

    def _save_scores(self, score_dct):
        print('--------------------SAVING SCORES--------------------')
        with open(os.path.join(self.cfg['result_path'], 'scores.pkl'), 'wb') as f:
            pickle.dump(score_dct, f)

    def _get_best_results(self, normal_scores, abnormal_scores, normal_freqs, abnormal_freqs, count_unk):
        assert type(normal_scores) == list
        assert type(abnormal_scores) == list
        assert type(normal_freqs) == list
        assert type(abnormal_freqs) == list

        scores = normal_scores + abnormal_scores
        freqs = normal_freqs + abnormal_freqs
        labels = [0] * len(normal_scores) + [1] * len(abnormal_scores)

        search_set = []
        for i in range(len(labels)):
            if labels[i] == 1:
                search_set.append([scores[i], freqs[i], True])
            else:
                search_set.append([scores[i], freqs[i], False])

        search_set.sort(key=lambda x: x[0])

        best_f1_res = -1
        tot_anomaly = sum(abnormal_freqs)
        threshold = 1
        P = 0 # P: point
        TP = 0
        best_P = 0
        best_TP = 0
        for i in range(len(search_set)):
            P += search_set[i][1]
            if search_set[i][2]:
                TP += search_set[i][1]
            precision = TP / (P + 1e-5)
            recall = TP / (tot_anomaly + 1e-5)
            f1 = 2 * precision * recall / (precision + recall + 1e-5)
            if f1 > best_f1_res:
                best_f1_res = f1
                threshold = search_set[i][0]
                best_P = P
                best_TP = TP

        f_TP = best_TP + count_unk
        f_FP = best_P - best_TP
        f_FN = tot_anomaly - best_TP
        f_TN = sum(freqs) - best_P - tot_anomaly + best_TP
        f_precision = f_TP / (f_TP + f_FP + 1e-5)
        f_recall = f_TP / (f_TP + f_FN + 1e-5)
        f_f1 = 2 * f_precision * f_recall / (f_precision + f_recall + 1e-5)

        return {
            'best_f1': best_f1_res,
            'u_best_f1': f_f1,
            'threshold': threshold,
            'precision': f_precision,
            'recall': f_recall,
            'TP': f_TP,
            'FP': f_FP,
            'FN': f_FN,
            'TN': f_TN
        }

    def delete_element(self, seq, n, u_idxs):
        seq = seq.copy()
        flag = 0
        for i in range(n):
            count = 0
            while count < 1000:
                pos = random.sample(range(0, len(seq)), 1)[0]
                if seq[pos] in u_idxs:
                    del seq[pos]
                    flag = 1
                    break
                count += 1

        return seq, flag

    def shuffle_element(self, seq, n, u_idxs):
        seq = seq.copy()
        flag = 0
        for _ in range(n):
            count = 0
            while count < 100:
                pos = random.sample(range(0, len(seq)), 2)
                if seq[pos[0]] in u_idxs and seq[pos[1]] in u_idxs and abs(pos[1] - pos[0]) < 3:
                    temp = seq[pos[0]]
                    seq[pos[0]] = seq[pos[1]]
                    seq[pos[1]] = temp
                    flag = 1
                    break
                count += 1
        return seq, flag

    def duplicate_element(self, seq, n, u_idxs):
        seq = seq.copy()
        flag = 0
        for _ in range(n):
            count = 0
            while count < 1000:
                pos = random.sample(range(0, len(seq)), 1)[0]
                if seq[pos] in u_idxs:
                    if pos == len(seq) - 1:
                        seq.append(seq[-1])
                    else:
                        last = seq[-1]
                        seq[pos + 1:] = seq[pos:-1]
                        seq.append(last)
                    flag = 1
                    break
                count += 1
        return seq, flag

    def data_augmentation(self, sessions, prob, n, length_limit=10, typ='delete', u_idxs=[]):
        print(sessions[0])
        sessions_idxs = []
        assert type(sessions) == list
        for idx, session in enumerate(sessions):
            if len(session) - n >= length_limit:
                sessions_idxs.append(idx)

        print('Number of sessions', len(sessions_idxs))
        print('Number of needed sessions', prob * len(sessions))
        

        assert (prob < 1 and prob > 0)
        if len(sessions_idxs) < prob * len(sessions):
            print('Warning: number of sessions that can be augmented is smaller than requirement')

        if typ == 'delete':
            augmentation_element = self.delete_element
        elif typ == 'shuffle':
            augmentation_element = self.shuffle_element
        else:
            augmentation_element = self.duplicate_element

        s_idxs = random.sample(sessions_idxs, int(prob * len(sessions)))

        count = 0
        for idx, s_idx in enumerate(s_idxs):
            sessions[s_idx], flag = augmentation_element(sessions[s_idx], n, u_idxs)
            count += flag

        print('Number of logs have been augmented', count)

        return sessions

    def _get_results_with_threshold(self, normal_scores, abnormal_scores, normal_freqs,
                                    abnormal_freqs, count_unk, threshold):
        assert type(normal_scores) == list
        assert type(abnormal_scores) == list
        assert type(normal_freqs) == list
        assert type(abnormal_freqs) == list

        scores = normal_scores + abnormal_scores
        freqs = normal_freqs + abnormal_freqs
        labels = [0] * len(normal_scores) + [1] * len(abnormal_scores)

        assert threshold != None

        TP = 0
        TN = 0
        FN = 0
        FP = 0
        for idx, score in enumerate(scores):
            if score >= threshold:
                if labels[idx] == 0:
                    TN += freqs[idx]
                else:
                    FN += freqs[idx]
            else:
                if labels[idx] == 1:
                    TP += freqs[idx]
                else:
                    FP += freqs[idx]

        TP += count_unk

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = precision * recall * 2 / (precision + recall)

        return {
            'best_f1': f1,
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN
        }

    def _calculate_results_with_threshold_set(self, threshold_set,
                                              normal_scores, abnormal_scores,
                                              normal_freqs, abnormal_freqs,
                                              count_unk):
        print('--------------------SAVING RESULTS WITH THRESHOLD SET--------------------')
        results = dict()
        for key, threshold in threshold_set.items():
            result_with_threshold = self._get_results_with_threshold(normal_scores, abnormal_scores,
                                                                     normal_freqs, abnormal_freqs,
                                                                     count_unk, threshold)
            results[key] = result_with_threshold

        with open(os.path.join(self.cfg['result_path'], 'results_with_thresholds.pkl'), 'wb') as f:
            pickle.dump(results, f)

    def _calculate_results_with_different_k(self, all_normal_scores, all_abnormal_scores,
                                            normal_sess_dct, non_unk_sess_dct, count_unk, generator_batch_size):
        print('--------------------SAVING RESULTS WITH DIFFERENT K--------------------')
        total_results = dict()
        for k in range(self.cfg['window_size']):
            normal_per_session, normal_freqs = self._calculate_scores_per_session(normal_sess_dct,
                                                                                       generator_batch_size,
                                                                                       all_normal_scores,
                                                                                       top_k=k+1)
            abnormal_per_session, abnormal_freqs = self._calculate_scores_per_session(non_unk_sess_dct,
                                                                                           generator_batch_size,
                                                                                           all_abnormal_scores,
                                                                                           top_k=k+1)
            results = self._get_best_results(normal_per_session,
                                                   abnormal_per_session,
                                                   normal_freqs,
                                                   abnormal_freqs,
                                                   count_unk)
            total_results[k+1] = {'best_f1': results['best_f1'],
                                  'u_best_f1': results['u_best_f1'],
                                  'precision': results['precision'],
                                  'recall': results['recall']}

        with open(os.path.join(self.cfg['result_path'], 'different_k_results.pkl'), 'wb') as f:
            pickle.dump(total_results, f)