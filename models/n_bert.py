import os
import sys
import numpy as np

HOME = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(HOME + '/preprocess')

from tqdm import tqdm
from base_model import BaseModel
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from embeddinglayer import BERTEmbedding
from transformer import TransformerEncoder
from model_utils import create_optimizer
from subsequencegenerator import SubsequenceGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

class NBert(BaseModel):
    def __init__(self, cfg, embedding_matrix, log_template_dict):
        super(NBert, self).__init__(cfg, embedding_matrix, log_template_dict)

    def _build_model(self):
        inputs = Input(shape=(self.cfg['window_size'],))
        embeddings = BERTEmbedding(vocab_size=self.cfg['vocab_size'],
                                   embed_size=self.cfg['reduction_dim'],
                                   window_size=self.cfg['window_size'],
                                   num_spec_tokens=self.cfg['num_spec_tokens'],
                                   embedding_matrix=self.embedding_matrix,
                                   dropout=self.cfg['dropout'])(inputs)
        trans_encoded = TransformerEncoder(num_blocks=self.cfg['num_blocks'],
                                           embed_size=self.cfg['reduction_dim'],
                                           num_heads=self.cfg['num_heads'],
                                           ff_dim=self.cfg['ff_dim'],
                                           window_size=self.cfg['window_size'],
                                           vocab_size=self.cfg['vocab_size'],
                                           dropout=self.cfg['dropout'])(embeddings)
        average_pooling = GlobalAveragePooling1D()(trans_encoded)
        output = Dense(self.cfg['vocab_size'], activation='softmax', name='next_out')(average_pooling)

        model = Model(inputs=inputs, outputs=output)
        self.model = model
        print(self.model.summary())

    def _compile(self, num_X_train):
        optimizer = create_optimizer(num_X_train=num_X_train,
                                     batch_size=self.cfg['batch_size'],
                                     epochs=self.cfg['epochs'])
        self.model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=optimizer)

    def _calculate_scores_all_sessions(self, sess_dct, generator_batch_size):
        sub_generator = SubsequenceGenerator(sess_dct,
                                             batch_size=generator_batch_size,
                                             window_size=self.cfg['window_size'])

        total_nexts = []

        for batch_data in tqdm(sub_generator.get_batches()):
            subseqs = []
            next_labels = []
            for i in range(len(batch_data)):
                subseqs += batch_data[i][0]
                next_labels += batch_data[i][1]
            next_labels = [self.log_template_dict[str(v)] if v != 0 else self.log_template_dict['UNK'] for v in next_labels]

            converted_subseqs = []
            for subseq in subseqs:
                converted_subseq = [self.log_template_dict[str(v)] if v != 0 else self.log_template_dict['UNK'] for v in subseq]
                converted_subseqs.append(converted_subseq)

            converted_subseqs = np.reshape(converted_subseqs, (len(subseqs), self.cfg['window_size']))
            converted_subseqs = np.float32(converted_subseqs)
            next_labels = to_categorical(next_labels, num_classes=self.cfg['vocab_size'])

            next_output = self.model.predict(converted_subseqs)

            # NEXT OUTPUT
            mask_next_labels = (next_labels == 1)
            next_output = next_output[mask_next_labels].reshape(-1, 1)
            total_nexts.append(next_output)

        return total_nexts

    def _load_model(self, model_file):
        model = load_model(os.path.join(self.cfg['result_path'], model_file), custom_objects={
            'TransformerEncoder': TransformerEncoder,
            'AdamWeightDecay': create_optimizer(),
            'BERTEmbedding': BERTEmbedding
        })
        return model

    def test(self, normal_sess_dct, abnormal_sess_dct, X_val=None, Y_val=None):
        self._load_model(self.cfg['model_file'])

        # Calculate scores per session
        print('Testing normal data')
        all_normal_scores = self._calculate_scores_all_sessions(normal_sess_dct, self.cfg['generator_batch_size'])
        print('Testing abnormal data')
        all_abnormal_scores = self._calculate_scores_all_sessions(abnormal_sess_dct, self.cfg['generator_batch_size'])

        normal_scores_per_session, normal_freqs = self._calculate_scores_per_session(normal_sess_dct,
                                                                                     self.cfg['generator_batch_size'],
                                                                                     all_normal_scores)
        abnormal_scores_per_session, abnormal_freqs = self._calculate_scores_per_session(abnormal_sess_dct,
                                                                                         self.cfg['generator_batch_size'],
                                                                                         all_abnormal_scores)

        # Get best result
        results = self._get_best_results(normal_scores_per_session,
                                         abnormal_scores_per_session,
                                         normal_freqs,
                                         abnormal_freqs)

        print('results: ', results)
        self._save_best_results(results)
        self._save_normal_abnormal_scores_plot(normal_scores_per_session,
                                               abnormal_scores_per_session,
                                               'Probs')
        score_dct = {'probs': {
            'normal': normal_scores_per_session,
            'abnormal': abnormal_scores_per_session
        }}
        self._save_scores(score_dct)

