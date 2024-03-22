import os
import sys
import numpy as np

HOME = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(HOME + '/preprocess')

from tqdm import tqdm
from base_model import BaseModel
from dataloader import DataLoader
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Reshape, Concatenate
from tensorflow.keras.models import Model, load_model
from subsequencegenerator import SubsequenceGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from collections import Counter
from model_utils import ShowLRate


class LogAnomaly(BaseModel):
    def __init__(self, cfg, embedding_matrix, log_template_dict):
        super(LogAnomaly, self).__init__(cfg=cfg,
                                         embedding_matrix=embedding_matrix,
                                         log_template_dict=log_template_dict)

    def _build_model(self):
        inputs1 = Input(shape=(self.cfg['window_size'], ))
        inputs2 = Input(shape=(self.cfg['vocab_size'], 1))

        if self.cfg['use_semantic']:
            embedding = Embedding(input_dim=self.cfg['vocab_size'],
                                  output_dim=self.cfg['reduction_dim'],
                                  weights=[self.embedding_matrix],
                                  trainable=False)
            embedded = embedding(inputs1)
        else:
            embedded = Reshape((self.cfg['window_size'], 1))(inputs1)

        output1 = LSTM(self.cfg['hidden_size'], return_sequences=True)(embedded)
        output1 = LSTM(self.cfg['hidden_size'], return_sequences=False)(output1)

        output2 = LSTM(self.cfg['hidden_size'], return_sequences=True)(inputs2)
        output2 = LSTM(self.cfg['hidden_size'], return_sequences=False)(output2)

        concat = Concatenate()([output1, output2])
        output = Dense(self.cfg['vocab_size'], activation='softmax')(concat)

        model = Model(inputs=[inputs1, inputs2], outputs=output)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics='accuracy')
        self.model = model
        print('Model summary', self.model.summary())

    def _scheduler(self, epoch, lr):
        if epoch == 0:
            lr = lr / 32
        elif epoch in [1, 2, 3, 4, 5]:
            lr = lr * 2
        elif epoch in [70, 90]:
            lr = lr * 0.1
        return lr

    def _load_model(self, model_file):
        print('----Loading model...')
        self.model = load_model(os.path.join(self.cfg['result_path'], model_file))

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
        lr_scheduler = LearningRateScheduler(self._scheduler)
        dataloader = DataLoader(X=X_train,
                                Y=Y_train,
                                batch_size=self.cfg['batch_size'],
                                vocab_size=self.cfg['vocab_size'],
                                log_template_dict=self.log_template_dict,
                                shuffle=True,
                                window_size=self.cfg['window_size'],
                                model_name=self.cfg['model_name'])

        history = self.model.fit(dataloader,
                                 epochs=self.cfg['epochs'],
                                 callbacks=[model_checkpoint, lr_scheduler, ShowLRate()],
                                 verbose=self.cfg['verbose'])

    def _calculate_scores_all_sessions(self, sess_dct, generator_batch_size):
        sub_generator = SubsequenceGenerator(sess_dct,
                                             batch_size=generator_batch_size,
                                             window_size=self.cfg['window_size'])
        total_scores = []
        freq_sess = []

        for batch_data in tqdm(sub_generator.get_batches()):
            subseqs = []
            next_labels = []
            for i in range(len(batch_data)):
                subseqs += batch_data[i][0]
                next_labels += batch_data[i][1]
            next_labels = [self.log_template_dict[str(v)] if v != 0 else self.log_template_dict['UNK'] for v in next_labels]

            features1 = []
            features2 = []
            for subseq in subseqs:
                mapping_subseq = [self.log_template_dict[str(v)] if v != 0 else self.log_template_dict['UNK'] for v in subseq]
                quantitative = [0] * self.cfg['vocab_size']
                log_counter = Counter(mapping_subseq)
                for key in log_counter:
                    try:
                        quantitative[key] = log_counter[key]
                    except:
                        pass

                features1.append(mapping_subseq)
                features2.append(quantitative)

            features1 = np.reshape(features1, (len(features1), self.cfg['window_size'], 1))
            features1 = np.float32(features1)

            features1 = np.array(features1).reshape(len(subseqs), self.cfg['window_size'])
            features2 = np.array(features2).reshape(len(subseqs), self.cfg['vocab_size'], 1)


            features1 = np.float32(features1)
            features2 = np.float32(features2)

            predictions = self.model.predict([features1, features2])
            next_labels = np.reshape(next_labels, (len(next_labels), 1))

            argsorted = np.argsort(predictions)
            scores = np.where(argsorted == next_labels)[1]

            for i in range(len(batch_data)):
                start_position = batch_data[i][2]
                end_position = batch_data[i][3]
                seq = batch_data[i][4]
                if start_position == end_position:
                    total_scores.append(np.min(scores[start_position]))
                else:
                    total_scores.append(np.min(scores[start_position:end_position]))
                freq_sess.append(sess_dct[seq])

        return total_scores, freq_sess

    def test(self, normal_sess_dct, abnormal_sess_dct, X_val=None, Y_val=None):
        self._load_model(self.cfg['model_file'])

        print('Testing normal data')
        normal_score_per_session, normal_freqs = self._calculate_scores_all_sessions(normal_sess_dct, self.cfg['generator_batch_size'])
        print('Test abnormal data')
        abnormal_score_per_session, abnormal_freqs = self._calculate_scores_all_sessions(abnormal_sess_dct, self.cfg['generator_batch_size'])

        next_results = self._get_best_results(normal_score_per_session,
                                              abnormal_score_per_session,
                                              normal_freqs,
                                              abnormal_freqs)
        print(next_results)
        self._save_best_results(next_results)

