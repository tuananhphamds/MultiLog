import os
import sys
import numpy as np

HOME = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(HOME + '/preprocess')

import tensorflow as tf
from base_model import BaseModel
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from embeddinglayer import BERTEmbedding
from transformer import TransformerEncoder
from model_utils import create_optimizer, custom_loss, custom_metric
from subsequencegenerator import SubsequenceGenerator
from tensorflow.keras.utils import to_categorical
from dataloader import ValDataLoader


class PNBert(BaseModel):
    def __init__(self, cfg, embedding_matrix, log_template_dict):
        super(PNBert, self).__init__(cfg, embedding_matrix, log_template_dict)

    def _build_model(self):
        inputs = Input(shape=(self.cfg['window_size'] + 1,))
        embeddings = BERTEmbedding(vocab_size=self.cfg['vocab_size'],
                                   embed_size=self.cfg['embed_size'],
                                   window_size=self.cfg['window_size'] + 1,
                                   num_spec_tokens=self.cfg['num_spec_tokens'],
                                   embedding_matrix=self.embedding_matrix,
                                   dropout=self.cfg['dropout'])(inputs)
        trans_encoded = TransformerEncoder(num_blocks=self.cfg['num_blocks'],
                                           embed_size=self.cfg['embed_size'],
                                           num_heads=self.cfg['num_heads'],
                                           ff_dim=self.cfg['ff_dim'],
                                           window_size=self.cfg['window_size'] + 1,
                                           vocab_size=self.cfg['vocab_size'],
                                           dropout=self.cfg['dropout'])(embeddings)
        output1 = Dense(self.cfg['vocab_size'], activation='softmax', name='mask_out')(trans_encoded)
        output2 = Dense(self.cfg['vocab_size'], activation='softmax', name='next_out')(trans_encoded[:, 0])

        model = Model(inputs=inputs, outputs=[output1, output2])
        self.model = model

    def _compile(self, num_X_train):
        cce = tf.keras.losses.CategoricalCrossentropy()
        acc_fn = tf.keras.metrics.Accuracy()
        optimizer = create_optimizer(num_X_train=num_X_train,
                                     batch_size=self.cfg['batch_size'],
                                     epochs=self.cfg['epochs'])
        self.model.compile(loss={'mask_out': custom_loss(cce=cce, vocab_size=self.cfg['vocab_size'], ignore_index=1),
                                 'next_out': 'categorical_crossentropy'},
                           metrics={'mask_out': custom_metric(acc_fn, vocab_size=self.cfg['vocab_size'], ignore_index=1),
                                    'next_out': 'acc'},
                           optimizer=optimizer,
                           loss_weights=self.cfg['loss_weights'])

    def _calculate_scores_all_sessions(self, sess_dct, generator_batch_size):
        sub_generator = SubsequenceGenerator(sess_dct,
                                             batch_size=generator_batch_size,
                                             window_size=self.cfg['window_size'])

        total_probs = []
        total_nexts = []

        for batch_data in sub_generator.get_batches():
            subseqs = []
            next_labels = []
            for i in range(len(batch_data)):
                subseqs += batch_data[i][0]
                next_labels += batch_data[i][1]
            next_labels = [self.log_template_dict[str(v)] if v != 0 else self.log_template_dict['UNK'] for v in next_labels]

            mask_subseqs = []
            mask_labels = []
            for subseq in subseqs:
                for i in range(self.cfg['window_size']):
                    new_subseq = [self.log_template_dict[str(v)] if v != 0 else self.log_template_dict['UNK'] for v in subseq]
                    label = [1] * self.cfg['window_size']
                    label[i] = new_subseq[i]
                    mask_labels.append([1] + label)
                    new_subseq[i] = self.log_template_dict['MASK']
                    mask_subseqs.append([self.log_template_dict['CLS']] + new_subseq)
            mask_subseqs = np.reshape(mask_subseqs, (len(mask_subseqs), self.cfg['window_size'] + 1))
            mask_subseqs = np.float32(mask_subseqs)
            mask_labels = np.reshape(mask_labels, (len(mask_labels), self.cfg['window_size'] + 1))
            next_labels = to_categorical(next_labels, num_classes=self.cfg['vocab_size'])

            mask_output, next_output = self.model.predict(mask_subseqs)

            # PROBS
            mask = (mask_labels != 1).reshape(-1, self.cfg['window_size'] + 1, 1)
            mask1 = np.tile(mask, (1, 1, self.cfg['vocab_size'])).astype(float)
            mask2 = to_categorical(mask_labels, num_classes=self.cfg['vocab_size'])
            mask3 = mask1 * mask2
            probs = mask_output[mask3 == 1].reshape(-1, self.cfg['window_size'])
            probs = np.sort(probs, axis=-1)

            # NEXT OUTPUT
            next_output = next_output.reshape(-1, self.cfg['window_size'], self.cfg['vocab_size'])
            next_labels = np.expand_dims(next_labels, axis=1)
            next_labels = np.tile(next_labels, (1, self.cfg['window_size'], 1))
            mask_next_labels = (next_labels == 1)
            next_output = next_output[mask_next_labels].reshape(-1, self.cfg['window_size'])
            next_output = np.sort(next_output, axis=-1)

            total_probs.append(probs)
            total_nexts.append(next_output)

        return total_probs, total_nexts

    def _load_model(self, model_file):
        print('----Loading model...')
        cce = tf.keras.losses.CategoricalCrossentropy()
        acc_fn = tf.keras.metrics.Accuracy()
        self.model = load_model(os.path.join(self.cfg['result_path'], model_file),
                                custom_objects={'TransformerEncoder': TransformerEncoder,
                                                              'AdamWeightDecay': create_optimizer(num_X_train=53053,
                                                                                                  batch_size=512,
                                                                                                  epochs=300),
                                                              'BERTEmbedding': BERTEmbedding,
                                                              'loss': custom_loss(cce=cce, vocab_size=self.cfg['vocab_size'],
                                                                                  ignore_index=1),
                                                              'acc': custom_metric(acc_fn=acc_fn, vocab_size=self.cfg['vocab_size'],
                                                                                   ignore_index=1)})

    def test(self, normal_sess_dct, abnormal_sess_dct, X_val=None, Y_val=None):
        self._load_model(self.cfg['model_file'])

        # normal_sessions = []
        # for sess, key in normal_sess_dct.items():
        #     normal_sessions += [list(sess)] * key
        # u_idxs = [2, 36, 44, 11, 41, 5]
        # results = self.data_augmentation(normal_sessions, 0.1, 1, length_limit=11, typ='delete', u_idxs=u_idxs)
        # normal_sess_dct = dict()
        # for sess in results:
        #     tuple_sess = tuple(sess)
        #     if tuple_sess not in normal_sess_dct:
        #         normal_sess_dct[tuple_sess] = 1
        #     else:
        #         normal_sess_dct[tuple_sess] += 1
        
        # print('Done')

        # Exclude sessions containing padding keys (0)
        non_unk_sess_dct = dict()
        count_unk = 0
        for key, value in abnormal_sess_dct.items():
            if 0 not in key:
                non_unk_sess_dct[key] = value
            else:
                count_unk += value

        generator_batch_size = 1024

        # Calculate scores per session
        all_normal_probs, all_normal_next = self._calculate_scores_all_sessions(normal_sess_dct, generator_batch_size)
        all_abnormal_probs, all_abnormal_next = self._calculate_scores_all_sessions(non_unk_sess_dct, generator_batch_size)

        # MASK PROBS
        normal_prob_per_session, normal_freqs = self._calculate_scores_per_session(normal_sess_dct,
                                                                                   generator_batch_size,
                                                                                   all_normal_probs,
                                                                                   top_k=self.cfg['top_k'])
        abnormal_probs_per_session, abnormal_freqs = self._calculate_scores_per_session(non_unk_sess_dct,
                                                                                       generator_batch_size,
                                                                                       all_abnormal_probs,
                                                                                       top_k=self.cfg['top_k'])

        # NEXT OUTPUT
        normal_next_per_session, normal_freqs = self._calculate_scores_per_session(normal_sess_dct,
                                                                                   generator_batch_size,
                                                                                   all_normal_next,
                                                                                   top_k=self.cfg['top_k'])
        abnormal_next_per_session, abnormal_freqs = self._calculate_scores_per_session(non_unk_sess_dct,
                                                                                       generator_batch_size,
                                                                                       all_abnormal_next,
                                                                                       top_k=self.cfg['top_k'])


        # SAVE BEST RESULTS
        next_results = self._get_best_results(normal_next_per_session,
                                         abnormal_next_per_session,
                                         normal_freqs,
                                         abnormal_freqs,
                                         count_unk)

        print('results: ', next_results)
        probs_results = self._get_best_results(normal_prob_per_session,
                                         abnormal_probs_per_session,
                                         normal_freqs,
                                         abnormal_freqs,
                                         count_unk)
        results = {
            'probs': probs_results,
            'next': next_results
        }
        self._save_best_results(results)

        # SAVE PLOTS
        self._save_normal_abnormal_scores_plot(normal_next_per_session,
                                               abnormal_next_per_session,
                                               'Next')
        self._save_normal_abnormal_scores_plot(normal_prob_per_session,
                                               abnormal_probs_per_session,
                                               'Probs')

        score_dct = {
            'probs': {
                'normal': normal_prob_per_session,
                'abnormal': abnormal_probs_per_session
            },
            'next': {
                'normal': normal_next_per_session,
                'abnormal': abnormal_next_per_session
            }
        }
        self._save_scores(score_dct)

        # # SAVE RESULTS WITH DIFFERENT K
        # self._calculate_results_with_different_k(all_normal_next, all_abnormal_next, normal_sess_dct,
        #                                          non_unk_sess_dct, count_unk, generator_batch_size)

        # # CALCULATE RESULTS WITH VALIDATION DATA
        # threshold_set = self._calculate_threshold_from_validation_data(X_val=X_val,
        #                                                                Y_val=Y_val,
        #                                                                top_k=self.cfg['top_k'],
        #                                                                batch_size=self.cfg['batch_size_val'])
        # self._calculate_results_with_threshold_set(threshold_set, normal_next_per_session, abnormal_next_per_session,
        #                                            normal_freqs, abnormal_freqs, count_unk)


    def _calculate_threshold_from_validation_data(self, X_val, Y_val, top_k=1, batch_size=64):
        total_scores = []

        val_dataloader = ValDataLoader(X=X_val,
                                       Y=Y_val,
                                       batch_size=batch_size,
                                       vocab_size=self.cfg['vocab_size'],
                                       log_template_dict=self.log_template_dict,
                                       shuffle=False,
                                       window_size=self.cfg['window_size'],
                                       hyper_center=self.hyper_center,
                                       model_name=self.cfg['model_name'])

        for batch in val_dataloader:
            batch_probs, batch_next = self.model.predict(batch[0])
            batch_next = batch_next.reshape(-1, self.cfg['window_size'], self.cfg['vocab_size'])
            batch_next_label = batch[1][1].reshape(-1, self.cfg['window_size'], self.cfg['vocab_size'])
            mask_next_label = batch_next_label == 1
            batch_next = batch_next[mask_next_label].reshape(-1, self.cfg['window_size'])
            batch_next = np.sort(batch_next, axis=-1)
            batch_next = batch_next[:, :top_k]
            batch_next = np.mean(batch_next, axis=-1)

            total_scores += batch_next.tolist()

        threshold_set = {
            '99': np.percentile(total_scores, 100 - 99),
            '99.5': np.percentile(total_scores, 100 - 99.5),
            '99.95': np.percentile(total_scores, 100 - 99.95),
            '99.99': np.percentile(total_scores, 100 - 99.99)
        }
        return threshold_set
