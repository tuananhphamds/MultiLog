import os
import sys
import numpy as np
import pickle

HOME = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(HOME + '/preprocess')

import tensorflow as tf
from base_model import BaseModel
from tensorflow.keras.layers import Input, Dense, LayerNormalization, GlobalAveragePooling1D 
from tensorflow.keras.models import Model, load_model
from embeddinglayer import BERTEmbedding
from identity import Identity
from transformer import TransformerEncoder
from subsequencegenerator import SubsequenceGenerator
from model_utils import create_optimizer, custom_metric, custom_loss, custom_contrast_loss
from tensorflow.keras.utils import to_categorical
from dataloader import ValDataLoader

class PNLBertCosine(BaseModel):
    def __init__(self, cfg, embedding_matrix, log_template_dict):
        super(PNLBertCosine, self).__init__(cfg, embedding_matrix, log_template_dict)

    def _build_model(self):
        inputs = Input(shape=(self.cfg['window_size'] + 2,))
        embeddings = BERTEmbedding(vocab_size=self.cfg['vocab_size'],
                                   embed_size=self.cfg['embed_size'],
                                   window_size=self.cfg['window_size'] + 2,
                                   num_spec_tokens=self.cfg['num_spec_tokens'],
                                   embedding_matrix=self.embedding_matrix,
                                   dropout=self.cfg['dropout'])(inputs)
        trans_encoded = TransformerEncoder(num_blocks=self.cfg['num_blocks'],
                                           embed_size=self.cfg['embed_size'],
                                           num_heads=self.cfg['num_heads'],
                                           ff_dim=self.cfg['ff_dim'],
                                           window_size=self.cfg['window_size'] + 2,
                                           vocab_size=self.cfg['vocab_size'],
                                           dropout=self.cfg['dropout'])(embeddings)
        output1 = Dense(self.cfg['vocab_size'], activation='softmax', name='mask_out')(trans_encoded[:, :-1])
        output2 = Dense(self.cfg['vocab_size'], activation='softmax', name='next_out')(trans_encoded[:, 0])

        sim_fn = tf.keras.losses.CosineSimilarity(reduction=tf.keras.losses.Reduction.NONE)
        average = GlobalAveragePooling1D()(trans_encoded[:, 1:-1])
        distance = 1 - tf.math.negative(sim_fn(trans_encoded[:, 0], average))
        output3 = Identity(name='contrast_out')(distance)

        model = Model(inputs=inputs, outputs=[output1, output2, output3])
        model.summary()
        self.model = model

    def _compile(self, num_X_train):
        cce = tf.keras.losses.CategoricalCrossentropy()
        acc_fn = tf.keras.metrics.Accuracy()
        optimizer = create_optimizer(num_X_train=num_X_train,
                                     batch_size=self.cfg['batch_size'],
                                     epochs=self.cfg['epochs'])
        self.model.compile(loss={'mask_out': custom_loss(cce=cce, vocab_size=self.cfg['vocab_size'], ignore_index=1),
                                 'next_out': 'categorical_crossentropy',
                                 'contrast_out': custom_contrast_loss(margin=1)},
                           metrics={
                               'mask_out': custom_metric(acc_fn, vocab_size=self.cfg['vocab_size'], ignore_index=1),
                               'next_out': 'acc',
                               'contrast_out': custom_contrast_loss(margin=1)},
                           optimizer=optimizer,
                           loss_weights=self.cfg['loss_weights'])

    def _calculate_scores_all_sessions(self, sess_dct, generator_batch_size):
        """
        # Test part
        dataset_path = os.path.join(self.base_path, 'datasets', self.cfg['dataset'])
        with open(os.path.join(dataset_path, 'template_similarities.pkl'), 'rb') as f:
            template_sims = pickle.load(f)
        with open(os.path.join(dataset_path, 'differences.pkl'), 'rb') as f:
            differences = pickle.load(f)
        #differences = [20, 27]
        sim_thresh = 0.9
        converted_sess_dct = dict()
        for key, value in sess_dct.items():
            new_key = []
            for id in key: 
                if id in differences and template_sims[id]['score'] >= sim_thresh:
                    new_key.append(template_sims[id]['template'])
                else:
                    new_key.append(id)
            if tuple(new_key) in converted_sess_dct:
                converted_sess_dct[tuple(new_key)] += value
            else:
                converted_sess_dct[tuple(new_key)] = value

        assert sum(converted_sess_dct.values()) == sum(sess_dct.values())
        """
        sub_generator = SubsequenceGenerator(sess_dct,
                                             batch_size=generator_batch_size)

        total_probs = []
        total_nexts = []
        total_scores = []

        for batch_data in sub_generator.get_batches():
            subseqs = []
            next_labels = []
            for i in range(len(batch_data)):
                subseqs += batch_data[i][0]
                next_labels += batch_data[i][1]
            next_labels = [self.log_template_dict[str(v)] if v != 0 else self.log_template_dict['UNK'] for v in next_labels]

            mask_subseqs = []
            mask_labels = []
            for idx, subseq in enumerate(subseqs):
                for i in range(self.cfg['window_size']):
                    new_subseq = [self.log_template_dict[str(v)] if v != 0 else self.log_template_dict['UNK'] for v in subseq]
                    label = [1] * self.cfg['window_size']
                    label[i] = new_subseq[i]
                    mask_labels.append([1] + label)
                    new_subseq[i] = self.log_template_dict['MASK']
                    mask_subseqs.append([self.log_template_dict['CLS']] + new_subseq + [next_labels[idx]])

            mask_subseqs = np.reshape(mask_subseqs, (len(mask_subseqs), self.cfg['window_size'] + 2))
            mask_subseqs = np.float32(mask_subseqs)
            mask_labels = np.reshape(mask_labels, (len(mask_labels), self.cfg['window_size'] + 1))
            next_labels = to_categorical(next_labels, num_classes=self.cfg['vocab_size'])

            mask_output, next_output, dists = self.model.predict(mask_subseqs)

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

            # CENTER HYPERSPHERE
            dists = 1 - dists
            dists = dists.reshape(-1, self.cfg['window_size'])
            dists = np.sort(dists, axis=-1)

            total_probs.append(probs)
            total_nexts.append(next_output)
            total_scores.append(dists)

        return total_probs, total_nexts, total_scores, sess_dct

    # def _extract_interested_seqs(self, sess_dct, generator_batch_size, type='normal', threshold=None):
    #     sub_generator = SubsequenceGenerator(sess_dct,
    #                                          batch_size=generator_batch_size)
    #
    #     filtered_seqs = []
    #
    #     for batch_data in sub_generator.get_batches():
    #         subseqs = []
    #         next_labels = []
    #         for i in range(len(batch_data)):
    #             subseqs += batch_data[i][0]
    #             next_labels += batch_data[i][1]
    #         next_labels = [self.log_template_dict[str(v)] if v != 0 else self.log_template_dict['UNK'] for v in next_labels]
    #
    #         mask_subseqs = []
    #         mask_labels = []
    #
    #         for idx, subseq in enumerate(subseqs):
    #             for i in range(self.cfg['window_size']):
    #                 new_subseq = [self.log_template_dict[str(v)] if v != 0 else self.log_template_dict['UNK'] for v in subseq]
    #                 label = [1] * self.cfg['window_size']
    #                 label[i] = new_subseq[i]
    #                 mask_labels.append([1] + label)
    #                 new_subseq[i] = self.log_template_dict['MASK']
    #                 mask_subseqs.append([self.log_template_dict['CLS']] + new_subseq + [next_labels[idx]])
    #
    #         mask_subseqs = np.reshape(mask_subseqs, (len(mask_subseqs), self.cfg['window_size'] + 2))
    #         mask_subseqs = np.float32(mask_subseqs)
    #         mask_labels = np.reshape(mask_labels, (len(mask_labels), self.cfg['window_size'] + 1))
    #         next_labels = to_categorical(next_labels, num_classes=self.cfg['vocab_size'])
    #
    #         batch_hyper_center = np.tile(self.hyper_center.reshape(1, 384), (len(mask_subseqs), 1))
    #         mask_output, next_output, dists = self.model.predict([mask_subseqs, batch_hyper_center])
    #
    #         # CENTER HYPERSPHERE
    #         dists = dists.reshape(-1, self.cfg['window_size'])
    #
    #         mask_subseqs = np.reshape(mask_subseqs, (len(subseqs), self.cfg['window_size'], -1))
    #
    #         for idx, value in enumerate(dists):
    #             sorted_value = np.sort(30 - value)
    #             if type == 'normal':
    #                 if np.mean(sorted_value[:5]) > threshold:
    #                     max_value = np.max(sorted_value)
    #                     index = np.where((30 - value) == max_value)[0][0]
    #                     filtered_seqs.append(mask_subseqs[idx][index].tolist())
    #             else:
    #                 if np.mean(sorted_value[:5] < threshold):
    #                     min_value = np.min(sorted_value)
    #                     index = np.where((30 - value) == min_value)[0][0]
    #                     filtered_seqs.append(mask_subseqs[idx][index].tolist())
    #
    #         if len(filtered_seqs) > 2000:
    #             break
    #
    #     filtered_seqs = np.reshape(filtered_seqs, (len(filtered_seqs), self.cfg['window_size'] + 2))
    #     cls_embeddings = self.feature_model.predict(filtered_seqs)
    #     return cls_embeddings

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
                                                                                   ignore_index=1),
                                                              'Identity': Identity,
                                                              'contrast_loss': custom_contrast_loss(30)})

    # def _feature_model(self):
    #     inputs = Input(shape=(self.cfg['window_size'] + 2,))
    #     embeddings_layer = self.model.get_layer(name='bert_embedding')
    #     transformer_encoder_layer = self.model.get_layer(name='transformer_encoder')
    #     embeddings = embeddings_layer(inputs)
    #     trans_encoded = transformer_encoder_layer(embeddings)
    #     cls_embedding = trans_encoded[:, 0]
    #     self.feature_model = Model(inputs=inputs, outputs=cls_embedding)

    # def test(self, normal_sess_dct, abnormal_sess_dct, X_val=None, Y_val=None):
    #     self._load_model(self.cfg['model_file'])
    #
    #     self._feature_model()
    #
    #     # Exclude sessions containing padding keys (0)
    #     non_unk_sess_dct = dict()
    #     count_unk = 0
    #     for key, value in abnormal_sess_dct.items():
    #         if 0 not in key:
    #             non_unk_sess_dct[key] = value
    #         else:
    #             count_unk += value
    #     generator_batch_size = 32
    #     normal_cls_embedding = self._extract_interested_seqs(normal_sess_dct, generator_batch_size, type='normal', threshold=14.81)
    #     abnormal_cls_embedding = self._extract_interested_seqs(non_unk_sess_dct, generator_batch_size, type='abnormal', threshold=14.81)
    #
    #     cls_embeddings = {
    #         'normal': normal_cls_embedding,
    #         'center': self.hyper_center.reshape(1, 384),
    #         'abnormal': abnormal_cls_embedding
    #     }
    #
    #     with open('cls_embeddings.pickle', 'wb') as f:
    #         pickle.dump(cls_embeddings, f)


    def test(self, normal_sess_dct, abnormal_sess_dct, X_val=None, Y_val=None):
        self._load_model(self.cfg['model_file'])
        print(self.model.summary())
        # normal_sessions = []
        # for sess, key in normal_sess_dct.items():
        #     normal_sessions += [list(sess)] * key
        # # u_idxs = [5, 26, 11, 9, 21, 23, 22]
        # u_idxs = [24, 19, 17, 1, 12, 29, 8, 15, 10,14 ,27, 28, 13,7, 16,25, 18, 6, 20, 2,4,3, 0]
        # results = self.data_augmentation(normal_sessions, 0.1, 2, length_limit=11, typ='delete', u_idxs=u_idxs)
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

        print('len non_unk_sess', len(non_unk_sess_dct))
        generator_batch_size = 32

        # Calculate scores per session
        all_normal_probs, all_normal_next, all_normal_dists, normal_sess_dct = self._calculate_scores_all_sessions(normal_sess_dct, generator_batch_size)

        all_abnormal_probs, all_abnormal_next, all_abnormal_dists, non_unk_sess_dct = self._calculate_scores_all_sessions(
            non_unk_sess_dct, generator_batch_size)
        print('Finished')


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

        # HYPERSPHERE
        normal_dist_per_session, normal_freqs = self._calculate_scores_per_session(normal_sess_dct,
                                                                                   generator_batch_size,
                                                                                   all_normal_dists,
                                                                                   top_k=self.cfg['top_k'])
        abnormal_dist_per_session, abnormal_freqs = self._calculate_scores_per_session(non_unk_sess_dct,
                                                                                       generator_batch_size,
                                                                                       all_abnormal_dists,
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

        dist_results = self._get_best_results(normal_dist_per_session,
                                               abnormal_dist_per_session,
                                               normal_freqs,
                                               abnormal_freqs,
                                               count_unk)

        results = {
            'next': next_results,
            'prob': probs_results,
            'dist': dist_results
        }
        self._save_best_results(results)

        threshold = 0.0025369222
        results = self._get_results_with_threshold(normal_next_per_session, abnormal_next_per_session, normal_freqs,
                                                   abnormal_freqs, count_unk, threshold)
        print('Augmentation results', results)

        # SAVE PLOTS
        self._save_normal_abnormal_scores_plot(normal_prob_per_session,
                                               abnormal_probs_per_session,
                                               'Prob')
        self._save_normal_abnormal_scores_plot(normal_next_per_session,
                                               abnormal_next_per_session,
                                               'Next')
        self._save_normal_abnormal_scores_plot(normal_dist_per_session,
                                               abnormal_dist_per_session,
                                               'Dist')
        score_dct = {
            'probs': {
                'normal': normal_prob_per_session,
                'abnormal': abnormal_probs_per_session
            },
            'next': {
                'normal': normal_next_per_session,
                'abnormal': abnormal_next_per_session
            },
            'dist': {
                'normal': normal_dist_per_session,
                'abnormal': abnormal_dist_per_session
            }
        }
        self._save_scores(score_dct)

        # SAVE RESULTS WITH DIFFERENT K
        # self._calculate_results_with_different_k(all_normal_next, all_abnormal_next, normal_sess_dct,
        #                                          non_unk_sess_dct, count_unk, generator_batch_size)
        #
        # # CALCULATE RESULTS WITH VALIDATION DATA
        # threshold_set = self._calculate_threshold_from_validation_data(X_val=X_val,
        #                                                                Y_val=Y_val,
        #                                                                top_k=self.cfg['top_k'],
        #                                                                batch_size=self.cfg['batch_size_val'])
        # self._calculate_results_with_threshold_set(threshold_set, normal_next_per_session, abnormal_next_per_session,
        #                                            normal_freqs, abnormal_freqs, count_unk)

    # def _calculate_threshold_from_validation_data(self, X_val, Y_val, top_k=1, batch_size=64):
    #     total_scores = []
    #
    #     val_dataloader = ValDataLoader(X=X_val,
    #                                    Y=Y_val,
    #                                    batch_size=batch_size,
    #                                    vocab_size=self.cfg['vocab_size'],
    #                                    log_template_dict=self.log_template_dict,
    #                                    shuffle=False,
    #                                    window_size=self.cfg['window_size'],
    #                                    hyper_center=self.hyper_center,
    #                                    predict_next_only=self.cfg['predict_next_only'])
    #
    #     for batch in val_dataloader:
    #         _, batch_next, _ = self.model.predict(batch[0])
    #         batch_next = batch_next.reshape(-1, self.cfg['window_size'], self.cfg['vocab_size'])
    #         batch_next_label = batch[1][1].reshape(-1, self.cfg['window_size'], self.cfg['vocab_size'])
    #         mask_next_label = batch_next_label == 1
    #         batch_next = batch_next[mask_next_label].reshape(-1, self.cfg['window_size'])
    #         batch_next = np.sort(batch_next, axis=-1)
    #         batch_next = batch_next[:, :top_k]
    #         batch_next = np.mean(batch_next, axis=-1)
    #
    #         total_scores += batch_next.tolist()
    #
    #     threshold_set = {
    #         '99': np.percentile(total_scores, 100 - 99),
    #         '99.5': np.percentile(total_scores, 100 - 99.5),
    #         '99.95': np.percentile(total_scores, 100 - 99.95),
    #         '99.99': np.percentile(total_scores, 100 - 99.99)
    #     }
    #     return threshold_set
