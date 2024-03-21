import numpy as np

class SubsequenceGenerator:
    """
    This class is to speed up the evaluation process
    """

    def __init__(self, seqs, window_size=10, batch_size=32):
        self.seqs = seqs
        self.batch_size = batch_size
        self.window_size = window_size
        self.all_subseqs = []
        self.__separate_sequence_into_subsequences()

    def __separate_sequence_into_subsequences(self):
        current_sequence_id = 0
        start_position = 0
        end_position = 0

        for seq in self.seqs:
            subseqs = []
            labels = []
            for i in range(len(seq) - self.window_size):
                subseq = seq[i:i + self.window_size]
                label = seq[i + self.window_size]
                labels.append(label)
                subseqs.append(subseq)
                end_position = end_position + 1
            self.all_subseqs.append((subseqs, labels, start_position, end_position, seq))
            start_position = end_position

            current_sequence_id = current_sequence_id + 1
            if current_sequence_id % self.batch_size == 0:
                start_position = 0
                end_position = 0

    def get_batches(self):
        num_batches = self.get_num_batches()
        for i in range(num_batches):
            yield self.all_subseqs[i * self.batch_size:(i + 1) * self.batch_size]

    def get_num_batches(self):
        if len(self.seqs) % self.batch_size == 0:
            return len(self.seqs) // self.batch_size
        return int(np.floor(len(self.seqs) / self.batch_size)) + 1
