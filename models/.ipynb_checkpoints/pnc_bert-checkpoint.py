import os
import sys
import numpy as np

HOME = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(HOME + '/preprocess')

import tensorflow as tf
from base_model import BaseModel
from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from embeddinglayer import BERTEmbedding
from identity import Identity
from transformer import TransformerEncoder
from subsequencegenerator import SubsequenceGenerator
from model_utils import create_optimizer, custom_metric, custom_loss, custom_contrast_loss
from tensorflow.keras.utils import to_categorical
from dataloader import ValDataLoader

class PNCBert(BaseModel):
    def __init__(self, cfg, embedding_matrix, log_template_dict):
        super(PNCBert, self).__init__(cfg, embedding_matrix, log_template_dict)

    # def _build_model(self):
    #     inputs = Input(shape=(self.cfg['window_size'] + 1,))
    #     embeddings = BERTEmbedding(vocab_size=self.cfg['vocab_size'],
    #                                embed_size=self.cfg['embed_size'],
    #                                window_size=self.cfg['window_size'] + 1,
    #                                num_spec_tokens=self.cfg['num_spec_tokens'],
    #                                embedding_matrix=self.embedding_matrix,
    #                                dropout=self.cfg['dropout'])(inputs)
    #     trans_encoded = TransformerEncoder(num_blocks=self.cfg['num_blocks'],
    #                                        embed_size=self.cfg['embed_size'],
    #                                        num_heads=self.cfg['num_heads'],
    #                                        ff_dim=self.cfg['ff_dim'],
    #                                        window_size=self.cfg['window_size'] + 1,
    #                                        vocab_size=self.cfg['vocab_size'],
    #                                        dropout=self.cfg['dropout'])(embeddings)
    #     output1 = Dense(self.cfg['vocab_size'], activation='softmax', name='mask_out')(trans_encoded)
    #     output2 = Dense(self.cfg['vocab_size'], activation='softmax', name='next_out')(trans_encoded[:, 0])
    #
    #     # Contrastive learning
    #     sim_fn = tf.keras.losses.CosineSimilarity(reduction=tf.keras.losses.Reduction.NONE)
    #     inputs1 = Input(shape=())
    #     target_embeddings = Embedding(input_dim=self.cfg['vocab_size'],
    #                                   output_dim=self.cfg['embed_size'],
    #                                   weights=[self.embedding_matrix],
    #                                   trainable=False)(inputs1)
    #     distances = 1 - tf.math.negative(sim_fn(target_embeddings, trans_encoded[:, 0]))
    #     output3 = Identity(name='contrast_out')(distances)
    #
    #     model = Model(inputs=[inputs, inputs1], outputs=[output1, output2, output3])
    #     self.model = model

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

        # Contrastive learning
        sim_fn = tf.keras.losses.CosineSimilarity(reduction=tf.keras.losses.Reduction.NONE)
        inputs1 = Input(shape=())
        target_embeddings = Embedding(input_dim=self.cfg['vocab_size'],
                                      output_dim=self.cfg['embed_size'],
                                      weights=[self.embedding_matrix],
                                      trainable=False)(inputs1)
        average = GlobalAveragePooling1D()(trans_encoded)
        distances = 1 - tf.math.negative(sim_fn(target_embeddings, average))
        output3 = Identity(name='contrast_out')(distances)

        model = Model(inputs=[inputs, inputs1], outputs=[output1, output2, output3])
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
        sub_generator = SubsequenceGenerator(sess_dct,
                                             batch_size=generator_batch_size)

        total_probs = []
        total_nexts = []
        total_dists = []

        for batch_data in sub_generator.get_batches():
            subseqs = []
            next_labels = []
            for i in range(len(batch_data)):
                subseqs += batch_data[i][0]
                next_labels += batch_data[i][1]
            next_labels = [self.log_template_dict[str(v)] if v != 0 else self.log_template_dict['UNK'] for v in next_labels]
            mask_next_labels = []
            for label in next_labels:
                mask_next_labels += [label] * self.cfg['window_size']


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
            mask_next_labels = np.array(mask_next_labels)

            mask_output, next_output, dists = self.model.predict([mask_subseqs, mask_next_labels])

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
            total_dists.append(dists)

        return total_probs, total_nexts, total_dists

    def test(self, normal_sess_dct, abnormal_sess_dct, X_val=None, Y_val=None):
        model = self._load_model(self.cfg['model_file'])

        # Exclude sessions containing padding keys (0)
        non_unk_sess_dct = dict()
        count_unk = 0
        for key, value in abnormal_sess_dct.items():
            if 0 not in key:
                non_unk_sess_dct[key] = value
            else:
                count_unk += value

        generator_batch_size = 32

        # Calculate scores per session
        all_normal_probs, all_normal_next, all_normal_dists = self._calculate_scores_all_sessions(normal_sess_dct, generator_batch_size)
        all_abnormal_probs, all_abnormal_next, all_abnormal_dists = self._calculate_scores_all_sessions(non_unk_sess_dct, generator_batch_size)

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

        # COSINE SIMILARITY
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

        dists_results = self._get_best_results(normal_dist_per_session,
                                               abnormal_dist_per_session,
                                               normal_freqs,
                                               abnormal_freqs,
                                               count_unk)

        results = {
            'next': next_results,
            'prob': probs_results,
            'dist': dists_results
        }
        self._save_best_results(results)

        # SAVE PLOTS
        self._save_normal_abnormal_scores_plot(normal_prob_per_session,
                                               abnormal_probs_per_session,
                                               'Prob')
        self._save_normal_abnormal_scores_plot(normal_next_per_session,
                                               abnormal_next_per_session,
                                               'Next')
        self._save_normal_abnormal_scores_plot(normal_dist_per_session,
                                               abnormal_dist_per_session,
                                               'Cosine similarity')
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
        self._calculate_results_with_different_k(all_normal_next, all_abnormal_next, normal_sess_dct,
                                                 non_unk_sess_dct, count_unk, generator_batch_size)

        # CALCULATE RESULTS WITH VALIDATION DATA
        threshold_set = self._calculate_threshold_from_validation_data(X_val=X_val,
                                                                       Y_val=Y_val,
                                                                       top_k=self.cfg['top_k'],
                                                                       batch_size=self.cfg['batch_size_val'])
        self._calculate_results_with_threshold_set(threshold_set, normal_next_per_session, abnormal_next_per_session,
                                                   normal_freqs, abnormal_freqs, count_unk)

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
            _, batch_next, _ = self.model.predict(batch[0])
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