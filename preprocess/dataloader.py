import numpy as np
import random
from tensorflow.keras.utils import Sequence, to_categorical
from collections import Counter


class DataLoader(Sequence):
    def __init__(self,
                 X,
                 Y,
                 batch_size=32,
                 vocab_size=32,
                 log_template_dict=None,
                 shuffle=True,
                 window_size=10,
                 hyper_center=None,
                 num_specs_token=4,
                 mask_rate=0.15,
                 model_name=None):
        self.X = X
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.shuffle = shuffle
        self.window_size = window_size
        self.log_template_dict = log_template_dict
        self.num_specs_token = num_specs_token
        self.mask_rate = mask_rate
        self.on_epoch_end()

        self.X = np.array(X).reshape(-1, self.window_size)
        self.X = np.int32(self.X)
        self.Y = np.array(Y)
        self.hyper_center = hyper_center

        if model_name == 'n_bert':
            self.__data_generation = self.__data_generation_next
        elif model_name == 'pn_bert':
            self.__data_generation = self.__data_generation_probs_next
        elif model_name in ['multilog']:
            self.__data_generation = self.__data_generation_probs_next_hyper
        elif model_name in ['logbert']:
            self.__data_generation = self.__data_generation_log_bert
        elif model_name in ['deeplog']:
            self.__data_generation = self.__data_generation_deeplog
        elif model_name in ['loganomaly']:
            self.__data_generation = self.__data_generation_loganomaly
            self._seq_counters = self._get_seq_counters(self.X)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.X) // self.batch_size + 1

    def __data_generation_next(self, indexes):
        """
        Generata batch_size samples
        """
        seqs = self.X[indexes]
        Y_batch = self.Y[indexes]
        converted_seqs = []
        labels = []
        for idx, seq in enumerate(seqs):
            converted_seq = [self.log_template_dict[str(v)] if v != 0 else self.log_template_dict['UNK'] for v in seq]
            converted_seqs.append(converted_seq)
            labels.append(self.log_template_dict[str(Y_batch[idx])])

        converted_seqs = np.array(converted_seqs).reshape(len(seqs), self.window_size)
        labels = to_categorical(labels, num_classes=self.vocab_size)

        return converted_seqs, labels

    def __data_generation_probs_next(self, indexes):
        """
        Generata batch_size samples
        """
        seqs = self.X[indexes]
        Y_batch = self.Y[indexes]
        random_seqs = []
        labels1 = []
        labels2 = []
        for idx, seq in enumerate(seqs):
            random_seq, label = self._random_template(seq)
            random_seq = [self.log_template_dict['CLS']] + random_seq
            label = [self.log_template_dict['PAD']] + label

            random_seqs.append(random_seq)
            labels1.append(label)
            labels2.append(self.log_template_dict[str(Y_batch[idx])])

        random_seqs = np.array(random_seqs).reshape(len(seqs), self.window_size + 1)
        labels1 = to_categorical(labels1, num_classes=self.vocab_size)
        labels2 = to_categorical(labels2, num_classes=self.vocab_size)

        return random_seqs, [labels1, labels2]

    def __data_generation_probs_next_hyper(self, indexes):
        """
        Generata batch_size samples
        """
        seqs = self.X[indexes]
        Y_batch = self.Y[indexes]
        random_seqs = []

        labels1 = []
        labels2 = []

        for idx, seq in enumerate(seqs):
            random_seq, label = self._random_template(seq, random_token=True)
            random_seq = [self.log_template_dict['CLS']] + random_seq
            label = [self.log_template_dict['PAD']] + label

            random_seqs.append(random_seq)

            labels1.append(label)
            labels2.append(self.log_template_dict[str(Y_batch[idx])])

        random_seqs = np.array(random_seqs).reshape(len(seqs), self.window_size + 1)

        labels1 = to_categorical(labels1, num_classes=self.vocab_size)
        labels2 = to_categorical(labels2, num_classes=self.vocab_size)

        return random_seqs, [labels1, labels2, self.hyper_center]

    def __data_generation_log_bert(self, indexes):
        """
        Generate batch_size samples
        """
        Y = self.Y[indexes].reshape(-1, 1)
        seqs = self.X[indexes]
        new_seqs = np.concatenate([seqs, Y], axis=1)
        random_seqs = []

        labels1 = []

        for idx, seq in enumerate(new_seqs):
            random_seq, label = self._random_template(seq, random_token=True)
            random_seq = [self.log_template_dict['CLS']] + random_seq

            label = [self.log_template_dict['PAD']] + label

            random_seqs.append(random_seq)

            labels1.append(label)

        random_seqs = np.array(random_seqs).reshape(len(seqs), self.window_size + 2)

        labels1 = to_categorical(labels1, num_classes=self.vocab_size)
        labels2 = np.zeros(len(seqs))  # labels2 is not used

        return random_seqs, [labels1, labels2]

    def __data_generation_deeplog(self, indexes):
        seqs = self.X[indexes]
        Y = self.Y[indexes]

        mapping_seqs = []
        mapping_Y = []
        for seq, y in zip(seqs, Y):
            mapping_seq = [self.log_template_dict[str(v)] if v != 0 else self.log_template_dict['UNK'] for v in seq]
            mapping_y = self.log_template_dict[str(y)] if y != 0 else self.log_template_dict['UNK']
            mapping_seqs.append(mapping_seq)
            mapping_Y.append(mapping_y)

        mapping_seqs = np.array(mapping_seqs).reshape(-1, self.window_size, 1)
        mapping_seqs = np.float32(mapping_seqs)
        mapping_Y = to_categorical(mapping_Y, num_classes=self.vocab_size)
        return mapping_seqs, mapping_Y

    def _get_seq_counters(self, X):
        seq_counters = dict()
        for seq in X:
            temp = [0] * self.vocab_size
            mapping_seq = [self.log_template_dict[str(v)] if v != 0 else self.log_template_dict['UNK'] for v in seq]
            counter = Counter(mapping_seq)
            for key in counter:
                try:
                    temp[key] = counter[key]
                except Exception as e:
                    pass
            seq_counters[tuple(seq)] = temp
        return seq_counters

    def __data_generation_loganomaly(self, indexes):
        seqs = self.X[indexes]
        Y = self.Y[indexes]

        features1 = []
        features2 = []
        labels = []
        for idx, seq in enumerate(seqs):
            feature1 = [self.log_template_dict[str(v)] if v != 0 else self.log_template_dict['UNK'] for v in seq]
            feature2 = self._seq_counters[tuple(seq)].copy()
            features1.append(feature1)
            features2.append(feature2)
            labels.append(self.log_template_dict[str(Y[idx])])

        features1 = np.array(features1).reshape(len(seqs), self.window_size)
        features2 = np.array(features2).reshape(len(seqs), self.vocab_size)
        labels = to_categorical(labels, num_classes=self.vocab_size)
        features1 = np.float32(features1)
        features2 = np.float32(features2)

        return [features1, features2], labels

    def _retrieve_negative_pairs(self, X, Y):
        positive_pairs = dict()
        for x, y in zip(X, Y):
            if x not in positive_pairs:
                positive_pairs[x] = set([y])
            else:
                positive_pairs[x].add(y)

        negative_pairs = dict()
        total_keys = set(range(1, self.vocab_size - self.num_specs_token + 1))
        for key, value in positive_pairs.items():
            negative_pairs[key] = list(total_keys - value)

        return negative_pairs

    def _get_random_log_key_target(self, seq, next_id):
        prob = random.random()
        negative_keys = self.negative_pairs[tuple(seq)]
        if prob > 0.5:
            return next_id, 1
        else:
            # Random key
            return self.log_template_dict[str(random.choice(negative_keys))], 0

    def _random_template(self, lk_seq, random_token=True):
        output_label = []
        output = []

        # 15% of the tokens would be replaced
        for i, lk in enumerate(lk_seq):
            prob = random.random()
            lk_id = self.log_template_dict[str(lk)]

            if prob < self.mask_rate:
                prob /= self.mask_rate

                # 80% chance change token to mask token
                if prob < 0.8:
                    output.append(self.log_template_dict['MASK'])

                # 10% chance change token to random token
                elif prob < 0.9 and random_token:
                    output.append(random.randrange(len(self.log_template_dict)))

                # 10% chance change token to current token
                else:
                    output.append(lk_id)
                output_label.append(lk_id)
            else:
                output.append(lk_id)
                output_label.append(1)

        assert len(output) == len(output_label)
        return output, output_label

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of indexes
        return self.__data_generation(indexes)