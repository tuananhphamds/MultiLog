import os
import sys
import numpy as np

HOME = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(HOME + '/preprocess')

from tqdm import tqdm
import tensorflow as tf
from base_model import BaseModel
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from embeddinglayer import BERTEmbedding
from identity import Identity
from transformer import TransformerEncoder
from subsequencegenerator import SubsequenceGenerator
from model_utils import create_optimizer, custom_metric, custom_loss, custom_distance
from tensorflow.keras.utils import to_categorical

class PNHBert(BaseModel):
    def __init__(self, cfg, embedding_matrix, log_template_dict, hyper_center):
        super(PNHBert, self).__init__(cfg, embedding_matrix, log_template_dict, hyper_center)

    def _build_model(self):
        inputs = Input(shape=(self.cfg['window_size'] + 1,))
        embeddings = BERTEmbedding(vocab_size=self.cfg['vocab_size'],
                                   embed_size=self.cfg['reduction_dim'],
                                   window_size=self.cfg['window_size'] + 1,
                                   num_spec_tokens=self.cfg['num_spec_tokens'],
                                   embedding_matrix=self.embedding_matrix,
                                   dropout=self.cfg['dropout'])(inputs)
        trans_encoded = TransformerEncoder(num_blocks=self.cfg['num_blocks'],
                                           embed_size=self.cfg['reduction_dim'],
                                           num_heads=self.cfg['num_heads'],
                                           ff_dim=self.cfg['ff_dim'],
                                           window_size=self.cfg['window_size'] + 1,
                                           vocab_size=self.cfg['vocab_size'],
                                           dropout=self.cfg['dropout'])(embeddings)
        output1 = Dense(self.cfg['vocab_size'], activation='softmax', name='mask_out')(trans_encoded)
        output2 = Dense(self.cfg['vocab_size'], activation='softmax', name='next_out')(trans_encoded[:, 0])
        output3 = Identity(name='hypersphere')(trans_encoded[:, 0])

        model = Model(inputs=inputs, outputs=[output1, output2, output3])
        self.model = model
        print('Model summary', self.model.summary())

    def _compile(self, num_X_train):
        cce = tf.keras.losses.CategoricalCrossentropy()
        acc_fn = tf.keras.metrics.Accuracy()
        optimizer = create_optimizer(num_X_train=num_X_train,
                                     batch_size=self.cfg['batch_size'],
                                     epochs=self.cfg['epochs'])
        self.model.compile(loss={'mask_out': custom_loss(cce=cce, vocab_size=self.cfg['vocab_size'], ignore_index=1),
                                 'next_out': 'categorical_crossentropy',
                                 'hypersphere': custom_distance},
                           metrics={
                               'mask_out': custom_metric(acc_fn, vocab_size=self.cfg['vocab_size'], ignore_index=1),
                               'next_out': 'acc',
                               'hypersphere': custom_distance},
                           optimizer=optimizer,
                           loss_weights=self.cfg['loss_weights'])

    def _calculate_scores_all_sessions(self, sess_dct, generator_batch_size):
        sub_generator = SubsequenceGenerator(sess_dct,
                                             batch_size=generator_batch_size,
                                            window_size=self.cfg['window_size'])

        total_probs = []
        total_nexts = []
        total_hypers = []

        for batch_data in tqdm(sub_generator.get_batches()):
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

            mask_output, next_output, temp_embed = self.model.predict(mask_subseqs)

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
            hypersphere_scores = np.linalg.norm(temp_embed - self.hyper_center.reshape(1, -1), axis=1)
            hypersphere_scores = hypersphere_scores.reshape(-1, self.cfg['window_size'])
            hypersphere_scores = -np.sort(-hypersphere_scores, axis=-1)

            total_probs.append(probs)
            total_nexts.append(next_output)
            total_hypers.append(hypersphere_scores)

        return total_probs, total_nexts, total_hypers

    def _load_model(self, model_file):
        print('----Loading model...')
        cce = tf.keras.losses.CategoricalCrossentropy()
        acc_fn = tf.keras.metrics.Accuracy()
        self.model = load_model(os.path.join(self.cfg['result_path'], model_file),
                                custom_objects={'TransformerEncoder': TransformerEncoder,
                                                              'AdamWeightDecay': create_optimizer(),
                                                              'BERTEmbedding': BERTEmbedding,
                                                              'loss': custom_loss(cce=cce, vocab_size=self.cfg['vocab_size'],
                                                                                  ignore_index=1),
                                                              'acc': custom_metric(acc_fn=acc_fn, vocab_size=self.cfg['vocab_size'],
                                                                                   ignore_index=1),
                                                              'Identity': Identity,
                                                              'custom_distance': custom_distance})

    def test(self, normal_sess_dct, abnormal_sess_dct, X_val=None, Y_val=None):
        self._load_model(self.cfg['model_file'])

        # Calculate scores per session
        print('Testing normal data')
        all_normal_probs, all_normal_next, all_normal_hyper = self._calculate_scores_all_sessions(normal_sess_dct, self.cfg['generator_batch_size'])
        print('Testing abnormal data')
        all_abnormal_probs, all_abnormal_next, all_abnormal_hyper = self._calculate_scores_all_sessions(abnormal_sess_dct, self.cfg['generator_batch_size'])

        # MASK PROBS
        normal_prob_per_session, normal_freqs = self._calculate_scores_per_session(normal_sess_dct,
                                                                                   self.cfg['generator_batch_size'],
                                                                                   all_normal_probs,
                                                                                   top_k=self.cfg['top_k'])
        abnormal_probs_per_session, abnormal_freqs = self._calculate_scores_per_session(abnormal_sess_dct,
                                                                                        self.cfg['generator_batch_size'],
                                                                                        all_abnormal_probs,
                                                                                        top_k=self.cfg['top_k'])

        # NEXT OUTPUT
        normal_next_per_session, normal_freqs = self._calculate_scores_per_session(normal_sess_dct,
                                                                                   self.cfg['generator_batch_size'],
                                                                                   all_normal_next,
                                                                                   top_k=self.cfg['top_k'])
        abnormal_next_per_session, abnormal_freqs = self._calculate_scores_per_session(abnormal_sess_dct,
                                                                                       self.cfg['generator_batch_size'],
                                                                                       all_abnormal_next,
                                                                                       top_k=self.cfg['top_k'])

        # HYPERSPHERE
        normal_hyper_per_session, normal_freqs = self._calculate_scores_per_session(normal_sess_dct,
                                                                                   self.cfg['generator_batch_size'],
                                                                                   all_normal_hyper,
                                                                                   top_k=self.cfg['top_k'],
                                                                                   min=False)
        abnormal_hyper_per_session, abnormal_freqs = self._calculate_scores_per_session(abnormal_sess_dct,
                                                                                       self.cfg['generator_batch_size'],
                                                                                       all_abnormal_hyper,
                                                                                       top_k=self.cfg['top_k'],
                                                                                       min=False)

        # SAVE BEST RESULTS
        next_results = self._get_best_results(normal_next_per_session,
                                         abnormal_next_per_session,
                                         normal_freqs,
                                         abnormal_freqs)

        print('results: ', next_results)
        probs_results = self._get_best_results(normal_prob_per_session,
                                               abnormal_probs_per_session,
                                               normal_freqs,
                                               abnormal_freqs)

        max_abnormal_hyper = max(abnormal_hyper_per_session)
        hyper_results = self._get_best_results((max_abnormal_hyper - np.array(normal_hyper_per_session)).tolist(),
                                               (max_abnormal_hyper - np.array(abnormal_hyper_per_session)).tolist(),
                                               normal_freqs,
                                               abnormal_freqs)

        results = {
            'next': next_results,
            'prob': probs_results,
            'hyper': hyper_results
        }
        
        self._save_best_results(results)

        # SAVE PLOTS
        self._save_normal_abnormal_scores_plot(normal_prob_per_session,
                                               abnormal_probs_per_session,
                                               'Prob')
        self._save_normal_abnormal_scores_plot(normal_next_per_session,
                                               abnormal_next_per_session,
                                               'Next')
        self._save_normal_abnormal_scores_plot(normal_hyper_per_session,
                                               abnormal_hyper_per_session,
                                               'Hyper')
        score_dct = {
            'probs': {
                'normal': normal_prob_per_session,
                'abnormal': abnormal_probs_per_session
            },
            'next': {
                'normal': normal_next_per_session,
                'abnormal': abnormal_next_per_session
            },
            'hyper': {
                'normal': normal_hyper_per_session,
                'abnormal': abnormal_hyper_per_session
            }
        }
        self._save_scores(score_dct)
        




