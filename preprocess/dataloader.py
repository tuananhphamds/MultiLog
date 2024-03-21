import numpy as np
import random
from tensorflow.keras.utils import Sequence, to_categorical


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
                 model_name=None,
                 reduction_dim=None):
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
        elif model_name in ['pnh_bert']:
            self.__data_generation = self.__data_generation_probs_next_hyper
        elif model_name in ['pnh_bert_euclidean']:
            self.hyper_center = hyper_center.reshape(1, reduction_dim)
            self.__data_generation = self.__data_generation_pnh_bert_euclidean
            self.negative_pairs = self._retrieve_negative_pairs(X, Y)
        elif model_name in ['pnl_bert_euclidean', 'pnl_bert_euclidean_average']:
            self.__data_generation = self.__data_generation_pnl_bert_euclidean
            self.negative_pairs = self._retrieve_negative_pairs(X, Y)

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
            random_seq, label = self._random_template(seq, random_token=True)
            random_seq = [self.log_template_dict['CLS']] + random_seq
            label = [self.log_template_dict['PAD']] + label

            random_seqs.append(random_seq)
            labels1.append(label)
            labels2.append(self.log_template_dict[str(Y_batch[idx])])

        random_seqs = np.array(random_seqs).reshape(len(seqs), self.window_size + 1)
        labels1 = to_categorical(labels1, num_classes=self.vocab_size)
        labels2 = to_categorical(labels2, num_classes=self.vocab_size)

        return random_seqs, [labels1, labels2]

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

    def __data_generation_pnh_bert_euclidean(self, indexes):
        seqs = self.X[indexes]
        Y_batch = self.Y[indexes]
        random_seqs = []
        inp_next_ids = []
        labels1 = []
        labels2 = []
        labels3 = []

        for idx, seq in enumerate(seqs):
            random_seq, label = self._random_template(seq, random_token=False)
            random_seq = [self.log_template_dict['CLS']] + random_seq
            label = [self.log_template_dict['PAD']] + label

            next_id = self.log_template_dict[str(Y_batch[idx])]
            inp_next_id, pair_label = self._get_random_log_key_target(seq, next_id)

            random_seqs.append(random_seq)
            inp_next_ids.append(inp_next_id)

            labels1.append(label)
            labels2.append(next_id)
            labels3.append(pair_label)

        random_seqs = np.array(random_seqs).reshape(len(seqs), self.window_size + 1)
        inp_next_ids = np.array(inp_next_ids).reshape(len(seqs), 1)

        labels1 = to_categorical(labels1, num_classes=self.vocab_size)
        labels2 = to_categorical(labels2, num_classes=self.vocab_size)
        labels3 = np.float32(labels3)

        X = np.concatenate([random_seqs, inp_next_ids], axis=-1)

        return [X, self.hyper_center], [labels1, labels2, labels3]

    def __data_generation_pnl_bert_euclidean(self, indexes):
        seqs = self.X[indexes]
        Y_batch = self.Y[indexes]
        random_seqs = []
        inp_next_ids = []
        labels1 = []
        labels2 = []
        labels3 = []

        for idx, seq in enumerate(seqs):
            random_seq, label = self._random_template(seq, random_token=False)
            random_seq = [self.log_template_dict['CLS']] + random_seq
            next_id = self.log_template_dict[str(Y_batch[idx])]
            label = [self.log_template_dict['PAD']] + label

            inp_next_id, pair_label = self._get_random_log_key_target(seq, next_id)

            random_seqs.append(random_seq)
            inp_next_ids.append(inp_next_id)

            labels1.append(label)
            labels2.append(next_id)
            labels3.append(pair_label)

        random_seqs = np.array(random_seqs).reshape(len(seqs), self.window_size + 1)
        inp_next_ids = np.array(inp_next_ids).reshape(len(seqs), 1)

        labels1 = to_categorical(labels1, num_classes=self.vocab_size)
        labels2 = to_categorical(labels2, num_classes=self.vocab_size)
        labels3 = np.float32(labels3)

        X = np.concatenate([random_seqs, inp_next_ids], axis=-1)

        return X, [labels1, labels2, labels3]

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

class ValDataLoader(Sequence):
    def __init__(self,
                 X,
                 Y,
                 batch_size=32,
                 vocab_size=32,
                 log_template_dict=None,
                 shuffle=False,
                 window_size=10,
                 hyper_center=None,
                 model_name=None):
        self.X = X
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.shuffle = shuffle
        self.window_size = window_size
        self.log_template_dict = log_template_dict
        self.on_epoch_end()

        self.X = np.array(X).reshape(-1, self.window_size)
        self.X = np.int32(self.X)
        self.Y = np.array(Y)
        self.hyper_center = hyper_center

        if model_name == 'n_bert':
            self.__data_generation = self.__data_generation_next
        elif model_name == 'pnc_bert':
            self.__data_generation = self.__data_generation_probs_next_dist
        elif model_name == 'pn_bert':
            self.__data_generation = self.__data_generation_probs_next
        elif model_name == 'pnh_bert':
            self.__data_generation = self.__data_generation_probs_next_hyper
        elif model_name == 'pncn_bert':
            self.__data_generation = self.__data_generation_probs_next_add_input

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of indexes
        return self.__data_generation(indexes)

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

    def _mask_template(self, lk_seq, idx):
        output = [self.log_template_dict[str(v)] if v != 0 else self.log_template_dict['UNK'] for v in lk_seq]
        output_label = [1] * self.window_size
        output_label[idx] = output[idx]
        output[idx] = self.log_template_dict['MASK']

        assert len(output) == len(output_label)
        return output, output_label

    def __data_generation_probs_next(self, indexes):
        """
        Generata batch_size samples
        """
        seqs = self.X[indexes]
        Y_batch = self.Y[indexes]
        mask_seqs = []
        labels1 = []
        labels2 = []
        for idx, seq in enumerate(seqs):
            for i in range(self.window_size):
                mask_seq, label = self._mask_template(seq, i)
                mask_seq = [self.log_template_dict['CLS']] + mask_seq
                label = [self.log_template_dict['PAD']] + label

                mask_seqs.append(mask_seq)
                labels1.append(label)
                labels2.append(self.log_template_dict[str(Y_batch[idx])])

        mask_seqs = np.array(mask_seqs).reshape(len(seqs) * self.window_size, self.window_size + 1)
        labels1 = to_categorical(labels1, num_classes=self.vocab_size)
        labels2 = to_categorical(labels2, num_classes=self.vocab_size)

        return mask_seqs, [labels1, labels2]

    def __data_generation_probs_next_dist(self, indexes):
        """
        Generata batch_size samples
        """
        seqs = self.X[indexes]
        Y_batch = self.Y[indexes]
        mask_seqs = []
        target_keys = []
        labels1 = []
        labels2 = []
        labels3 = []
        for idx, seq in enumerate(seqs):
            for i in range(self.window_size):
                mask_seq, label = self._mask_template(seq, i)
                mask_seq = [self.log_template_dict['CLS']] + mask_seq
                label = [self.log_template_dict['PAD']] + label

                mask_seqs.append(mask_seq)
                labels1.append(label)
                labels2.append(self.log_template_dict[str(Y_batch[idx])])

                target, pair_label = self.log_template_dict[str(Y_batch[idx])], 1
                target_keys.append(target)
                labels3.append(pair_label)

        mask_seqs = np.array(mask_seqs).reshape(len(seqs) * self.window_size, self.window_size + 1)
        labels1 = to_categorical(labels1, num_classes=self.vocab_size)
        labels2 = to_categorical(labels2, num_classes=self.vocab_size)
        labels3 = np.float32(labels3)
        target_keys = np.float32(target_keys)

        return [mask_seqs, target_keys], [labels1, labels2, labels3]

    def __data_generation_probs_next_add_input(self, indexes):
        seqs = self.X[indexes]
        Y_batch = self.Y[indexes]
        mask_seqs = []
        inp_next_ids = []
        labels1 = []
        labels2 = []
        labels3 = []

        for idx, seq in enumerate(seqs):
            for i in range(self.window_size):
                mask_seq, label = self._mask_template(seq, i)
                mask_seq = [self.log_template_dict['CLS']] + mask_seq
                label = [self.log_template_dict['PAD']] + label
                inp_next_id, pair_label = self.log_template_dict[str(Y_batch[idx])], 1

                mask_seqs.append(mask_seq)
                inp_next_ids.append(inp_next_id)
                labels1.append(label)
                labels2.append(self.log_template_dict[str(Y_batch[idx])])
                labels3.append(pair_label)

        mask_seqs = np.array(mask_seqs).reshape(len(seqs) * self.window_size, self.window_size + 1)
        labels1 = to_categorical(labels1, num_classes=self.vocab_size)
        labels2 = to_categorical(labels2, num_classes=self.vocab_size)
        labels3 = np.float32(labels3)
        inp_next_ids = np.array(inp_next_ids).reshape(len(seqs) * self.window_size, 1)

        X = np.concatenate([mask_seqs, inp_next_ids], axis=-1)

        return X, [labels1, labels2, labels3]


    def __data_generation_probs_next_hyper(self, indexes):
        """
        Generata batch_size samples
        """
        seqs = self.X[indexes]
        Y_batch = self.Y[indexes]
        mask_seqs = []
        labels1 = []
        labels2 = []
        for idx, seq in enumerate(seqs):
            for i in range(self.window_size):
                mask_seq, label = self._mask_template(seq, i)
                mask_seq = [self.log_template_dict['CLS']] + mask_seq
                label = [self.log_template_dict['PAD']] + label

                mask_seqs.append(mask_seq)
                labels1.append(label)
                labels2.append(self.log_template_dict[str(Y_batch[idx])])

        mask_seqs = np.array(mask_seqs).reshape(len(seqs) * self.window_size, self.window_size + 1)
        labels1 = to_categorical(labels1, num_classes=self.vocab_size)
        labels2 = to_categorical(labels2, num_classes=self.vocab_size)

        return mask_seqs, [labels1, labels2, self.hyper_center]


