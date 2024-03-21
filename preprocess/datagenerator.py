import os
import pickle
import random

from utils import check_dataset

HOME = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]

class DataGenerator:
    def __init__(self,
                 dataset,
                 window_size,
                 train_val_test=(1, 1, 98),
                 val_pos='head',
                 shuffle=True,
                 random_seed=50,
                 fixed_window=20):
        if len(train_val_test) < 3 or train_val_test[0] + train_val_test[1] + train_val_test[2] > 100:
            raise ValueError('Train, validation, and test ratio is invalid')
        assert check_dataset(dataset) == True

        self.train_rate, self.val_rate, self.test_rate = train_val_test
        self.dataset = dataset
        self.window_size = window_size
        self.val_pos = val_pos
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.fixed_window = fixed_window

        self.dataset_path = os.path.join(HOME, 'datasets', dataset)
        if 'bgl' in self.dataset or 'tbird' in self.dataset:
            self._generate_block_info()
        # Load data
        with open(os.path.join(self.dataset_path, 'block_info.pkl'), 'rb') as f:
            self.block_info = pickle.load(f)

    def _sliding_window(self, sessions, label=None):
        X = []
        Y = []
        for sess in sessions:
            line = [v for v in sess]
            line = line + [0] * (self.window_size - len(line) + 1)
            line = tuple(line)
            for i in range(len(line) - self.window_size):
                X.append(line[i:i + self.window_size])
                Y.append(line[i + self.window_size])

        print('Number of sessions({}): {}'.format(label, len(sessions)))
        print('Number of sequences({}): {}'.format(label, len(X)))
        return X, Y

    def _generate_block_info(self):
        with open(os.path.join(self.dataset_path, 'key_labels.pkl'), 'rb') as f:
            key_labels = pickle.load(f)
        assert key_labels is not None
        keys = key_labels['key']
        labels = key_labels['label']

        block_info = dict()
        log_size = len(keys)
        i = 0
        count = 1
        while i < log_size:
            key_ids = keys[i:i+self.fixed_window]
            seq_label = max(labels[i:i+self.fixed_window])
            block_info[str(count)] = {
                'seq': key_ids,
                'label': seq_label
            }
            i = i + self.fixed_window
            count += 1

        with open(os.path.join(self.dataset_path, 'block_info.pkl'), 'wb') as f:
            pickle.dump(block_info, f)

    def _get_session_dict(self, sessions, label=None):
        sess_dct = dict()
        for sess in sessions:
            line = [v for v in sess]
            line = line + [0] * (self.window_size - len(line) + 1)
            line = tuple(line)
            if line not in sess_dct:
                sess_dct[line] = 1
            else:
                sess_dct[line] += 1
        print('Number of unique sessions({}): {}'.format(label, len(sess_dct)))
        print('Number of sessions({}): {}'.format(label, len(sessions)))
        return sess_dct

    # def generate_train_val_test(self, anomaly_pct=0.01):
    #     normal_sessions = []
    #     abnormal_sessions = []
    #
    #     # Get normal, abnormal sessions
    #     for blk, info in self.block_info.items():
    #         if info['label'] == 0:
    #             normal_sessions.append(info['seq'])
    #         else:
    #             abnormal_sessions.append(info['seq'])
    #
    #     # Shuffle data
    #     if self.shuffle:
    #         random.seed(self.random_seed)
    #         random.shuffle(normal_sessions)
    #         random.shuffle(abnormal_sessions)
    #
    #     # Calculate number of sessions for train, val, and test
    #     total_normal_sessions = len(normal_sessions)
    #     n_train = int(total_normal_sessions * self.train_rate / 100)
    #     n_val = int(total_normal_sessions * self.val_rate / 100)
    #     n_test = int(total_normal_sessions * self.test_rate / 100)
    #
    #     train_sessions = normal_sessions[:n_train]
    #     test_sessions = normal_sessions[-n_test:]
    #     if self.val_pos == 'head':
    #         val_sessions = normal_sessions[n_train:n_train + n_val]
    #     else:
    #         val_sessions = normal_sessions[-n_test - n_val:-n_test]
    #
    #
    #     # Insert anomalies to training data
    #     first_abnormal_sessions = []
    #     second_abnormal_sessions = []
    #     for sess in abnormal_sessions:
    #         if len(sess) <= self.window_size:
    #             second_abnormal_sessions.append(sess)
    #         else:
    #             first_abnormal_sessions.append(sess)
    #     total_first_anomalies = len(first_abnormal_sessions)
    #     n_insert = int(total_first_anomalies * anomaly_pct)
    #     insert_sessions = first_abnormal_sessions[:n_insert]
    #     print('Number of inserted abnormal sessions', len(insert_sessions))
    #
    #
    #     # Get X_train, Y_train, X_val, Y_val, normal_test_sessions, abnormal_test_sessions
    #     X_train, Y_train = self._sliding_window(train_sessions + insert_sessions, 'train')
    #     X_val, Y_val = self._sliding_window(val_sessions, 'val')
    #
    #     # Speed up test phrase using unique sessions
    #     test_sessions_dct = self._get_session_dict(test_sessions, 'normal_test')
    #     abnormal_sessions_dct = self._get_session_dict(first_abnormal_sessions[n_insert:] + second_abnormal_sessions, 'abnormal_test')
    #
    #     return {
    #         'train': [X_train, Y_train],
    #         'val': [X_val, Y_val],
    #         'test': [test_sessions_dct, abnormal_sessions_dct]
    #     }

    def generate_train_val_test(self):
        normal_sessions = []
        abnormal_sessions = []

        # Get normal, abnormal sessions
        for blk, info in self.block_info.items():
            if info['label'] == 0:
                normal_sessions.append(info['seq'])
            else:
                abnormal_sessions.append(info['seq'])

        print('\nNumber of normal sessions', len(normal_sessions))
        print('\nNumber of abnormal sessions', len(abnormal_sessions))
        # Shuffle data
        if self.shuffle:
            random.seed(self.random_seed)
            random.shuffle(normal_sessions)

        # Calculate number of sessions for train, val, and test
        total_normal_sessions = len(normal_sessions)
        n_train = int(total_normal_sessions * self.train_rate / 100)
        n_val = int(total_normal_sessions * self.val_rate / 100)
        n_test = int(total_normal_sessions * self.test_rate / 100)

        train_sessions = normal_sessions[:n_train]
        test_sessions = normal_sessions[-n_test:]
        if self.val_pos == 'head':
            val_sessions = normal_sessions[n_train:n_train + n_val]
        else:
            val_sessions = normal_sessions[-n_test - n_val:-n_test]

        # Get X_train, Y_train, X_val, Y_val, normal_test_sessions, abnormal_test_sessions
        X_train, Y_train = self._sliding_window(train_sessions, 'train')
        X_val, Y_val = self._sliding_window(val_sessions, 'val')

        # Speed up test phrase using unique sessions
        test_sessions_dct = self._get_session_dict(test_sessions, 'normal_test')
        abnormal_sessions_dct = self._get_session_dict(abnormal_sessions, 'abnormal_test')

        return {
            'train': [X_train, Y_train],
            'val': [X_val, Y_val],
            'test': [test_sessions_dct, abnormal_sessions_dct]
        }

    # def generate_train_val_test(self):
    #     normal_sessions = []
    #     abnormal_sessions = []
    #
    #     # Get normal, abnormal sessions
    #     for blk, info in self.block_info.items():
    #         if info['label'] == 0:
    #             normal_sessions.append(info['seq'])
    #         else:
    #             abnormal_sessions.append(info['seq'])
    #
    #
    #     # Calculate number of sessions for train, val, and test
    #     total_normal_sessions = len(normal_sessions)
    #     n_train = int(total_normal_sessions * self.train_rate / 100)
    #     n_val = int(total_normal_sessions * self.val_rate / 100)
    #     n_test = int(total_normal_sessions * self.test_rate / 100)
    #
    #     test_sessions = normal_sessions[-n_test:]
    #     normal_sessions = normal_sessions[:(total_normal_sessions-n_test)]
    #
    #     # Shuffle data
    #     if self.shuffle:
    #         random.seed(self.random_seed)
    #         random.shuffle(normal_sessions)
    #
    #     train_sessions = normal_sessions[:n_train]
    #
    #     if self.val_pos == 'head':
    #         val_sessions = normal_sessions[n_train:n_train + n_val]
    #     else:
    #         val_sessions = normal_sessions[-n_val:]
    #
    #     # Get X_train, Y_train, X_val, Y_val, normal_test_sessions, abnormal_test_sessions
    #     X_train, Y_train = self._sliding_window(train_sessions, 'train')
    #     X_val, Y_val = self._sliding_window(val_sessions, 'val')
    #
    #     # Speed up test phrase using unique sessions
    #     test_sessions_dct = self._get_session_dict(test_sessions, 'normal_test')
    #     abnormal_sessions_dct = self._get_session_dict(abnormal_sessions, 'abnormal_test')
    #
    #     return {
    #         'train': [X_train, Y_train],
    #         'val': [X_val, Y_val],
    #         'test': [test_sessions_dct, abnormal_sessions_dct]
    #     }
