import os
import sys
import numpy as np
import pickle

HOME = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(HOME + '/preprocess')

import tensorflow as tf
from base_model import BaseModel
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model, load_model
from embeddinglayer import BERTEmbedding
from identity import Identity
from transformer import TransformerEncoder
from subsequencegenerator import SubsequenceGenerator
from model_utils import create_optimizer, custom_metric, custom_loss, custom_contrast_loss
from tensorflow.keras.utils import to_categorical

class ContrastiveCenter(BaseModel):
    def __init__(self, cfg, embedding_matrix, log_template_dict, hyper_center):
        super(ContrastiveCenter, self).__init__(cfg, embedding_matrix, log_template_dict, hyper_center)

    def _build_model(self):
        inputs = Input(shape=(self.cfg['window_size'] + 1,))
        inputs1 = Input(shape=(self.cfg['embed_size'],))
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
        sequence_average = GlobalAveragePooling1D()(trans_encoded)

        # Contrastive learning
        sim_fn = tf.keras.losses.CosineSimilarity(reduction=tf.keras.losses.Reduction.NONE)
        distances = 1 - tf.math.negative(sim_fn(sequence_average, inputs1))
        output = Identity(name='contrast_out')(distances)

        model = Model(inputs=[inputs, inputs1], outputs=output)
        self.model = model

    def _compile(self, num_X_train):
        optimizer = create_optimizer(num_X_train=num_X_train,
                                     batch_size=self.cfg['batch_size'],
                                     epochs=self.cfg['epochs'])
        self.model.compile(loss=custom_contrast_loss(margin=1),
                           metrics=custom_contrast_loss(margin=1),
                           optimizer=optimizer)

    def _calculate_scores_all_sessions(self, sess_dct, generator_batch_size):
        sub_generator = SubsequenceGenerator(sess_dct,
                                             batch_size=generator_batch_size)

        total_scores = []

        for batch_data in sub_generator.get_batches():
            subseqs = []
            next_labels = []
            for i in range(len(batch_data)):
                subseqs += batch_data[i][0]
                next_labels += batch_data[i][1]
            next_labels = [self.log_template_dict[str(v)] if v != 0 else self.log_template_dict['UNK'] for v in next_labels]

            new_subseqs = []
            for subseq in subseqs:
                new_subseq = [self.log_template_dict[str(v)] if v != 0 else self.log_template_dict['UNK'] for v in subseq]
                new_subseqs.append(new_subseq)

            new_subseqs = np.reshape(new_subseqs, (len(new_subseqs), self.cfg['window_size']))
            new_subseqs = np.float32(new_subseqs)
            inp_next_ids = np.reshape(next_labels, (len(next_labels), 1))
            inp_next_ids = np.float32(inp_next_ids)

            inputs = np.concatenate([new_subseqs, inp_next_ids], axis=-1)

            batch_hyper_center = np.tile(self.hyper_center.reshape(1, 384), (len(inp_next_ids), 1))

            dists = self.model.predict([inputs, batch_hyper_center])
            dists = 1 - dists
            total_scores.append(dists)

        return total_scores

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


    def test(self, normal_sess_dct, abnormal_sess_dct, X_val=None, Y_val=None):
        self._load_model(self.cfg['model_file'])

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
        all_normal_dists = self._calculate_scores_all_sessions(normal_sess_dct, generator_batch_size)

        all_abnormal_dists = self._calculate_scores_all_sessions(non_unk_sess_dct, generator_batch_size)
        print('Finished')


        # HYPERSPHERE
        normal_dist_per_session, normal_freqs = self._calculate_scores_per_session(normal_sess_dct,
                                                                                   generator_batch_size,
                                                                                   all_normal_dists,
                                                                                   mask=False)
        abnormal_dist_per_session, abnormal_freqs = self._calculate_scores_per_session(non_unk_sess_dct,
                                                                                       generator_batch_size,
                                                                                       all_abnormal_dists,
                                                                                       mask=False)

        # SAVE BEST RESULTS

        dist_results = self._get_best_results(normal_dist_per_session,
                                               abnormal_dist_per_session,
                                               normal_freqs,
                                               abnormal_freqs,
                                               count_unk)

        results = {
            'dist': dist_results
        }
        self._save_best_results(results)

        # SAVE PLOTS
        self._save_normal_abnormal_scores_plot(normal_dist_per_session,
                                               abnormal_dist_per_session,
                                               'Dist')
        score_dct = {
            'dist': {
                'normal': normal_dist_per_session,
                'abnormal': abnormal_dist_per_session
            }
        }
        self._save_scores(score_dct)

