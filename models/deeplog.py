import os
import sys
import numpy as np

HOME = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(HOME + '/preprocess')

from tqdm import tqdm
from base_model import BaseModel
from dataloader import DataLoader
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Sequential, load_model
from subsequencegenerator import SubsequenceGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from model_utils import ShowLRate

class DeepLog(BaseModel):
    def __init__(self, cfg, log_template_dict):
        super(DeepLog, self).__init__(cfg=cfg,
                                      log_template_dict=log_template_dict)

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.cfg['window_size'], 1)))
        model.add(LSTM(self.cfg['hidden_size'], return_sequences=True))
        model.add(LSTM(self.cfg['hidden_size'], return_sequences=False))
        model.add(Dense(self.cfg['vocab_size'], activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        self.model = model
        print('Model summary', self.model.summary())

    def _scheduler(self, epoch, lr):
        if epoch == 0:
            lr = lr / 32
        elif epoch in [1, 2, 3, 4, 5]:
            lr = lr * 2
        elif epoch in [45, 47]:
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

            mapping_subseqs = []
            for subseq in subseqs:
                mapping_subseq = [self.log_template_dict[str(v)] if v != 0 else self.log_template_dict['UNK'] for v in subseq]
                mapping_subseqs.append(mapping_subseq)

            mapping_subseqs = np.reshape(mapping_subseqs, (len(mapping_subseqs), self.cfg['window_size'], 1))
            mapping_subseqs = np.float32(mapping_subseqs)

            predictions = self.model.predict(mapping_subseqs)
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
        print('Testing abnormal data')
        abnormal_score_per_session, abnormal_freqs = self._calculate_scores_all_sessions(abnormal_sess_dct, self.cfg['generator_batch_size'])

        next_results = self._get_best_results(normal_score_per_session,
                                              abnormal_score_per_session,
                                              normal_freqs,
                                              abnormal_freqs)
        print(next_results)
        self._save_best_results(next_results)
        
