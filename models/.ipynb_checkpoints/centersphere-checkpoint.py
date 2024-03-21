import numpy as np
from tensorflow.keras.layers import Input, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from embeddinglayer import LogTemplateEmbedding

class CenterSphere:
    def __init__(self,
                 embedding_matrix,
                 num_vocab,
                 window_size,
                 embed_size,
                 num_spec_tokens,
                 log_template_dict):
        self.embedding_matrix = embedding_matrix
        self.num_vocab = num_vocab
        self.window_size = window_size
        self.embed_size = embed_size
        self.num_spec_tokens = num_spec_tokens
        self.log_template_dict = log_template_dict
        self.model = None

    def _build_template_model(self):
        model = Sequential()
        model.add(Input(shape=(self.window_size,)))
        model.add(LogTemplateEmbedding(vocab_size=self.num_vocab,
                                       embed_size=self.embed_size,
                                       num_spec_tokens=self.num_spec_tokens,
                                       embedding_matrix=self.embedding_matrix))
        model.add(GlobalAveragePooling1D())
        return model

    def calculate_center_sphere(self, log_sequences):
        if self.model is None:
            self.model = self._build_template_model()

        unique_seqs = set()
        for seq in log_sequences:
            convert_seq = [self.log_template_dict[str(v)] if v != 0 else self.log_template_dict['UNK'] for v in seq]
            unique_seqs.add(tuple(convert_seq))
        unique_seqs = list(unique_seqs)
        unique_seqs = np.array(unique_seqs)

        num_samples = 0
        center = np.zeros(self.embed_size)
        embeddings = self.model.predict(unique_seqs)
        center = np.sum(embeddings, axis=0)
        num_samples += embeddings.shape[0]
        hyper_center = center / num_samples
        return hyper_center

    # def calculate_center_sphere(self, log_sequences, batch_size=512):
    #     if self.model is None:
    #         self.model = self._build_template_model()

    #     total_seqs = list()
    #     for seq in log_sequences:
    #         convert_seq = [self.log_template_dict[str(v)] if v != 0 else self.log_template_dict['UNK'] for v in seq]
    #         total_seqs.append(tuple(convert_seq))
        
    #     total_seqs = np.array(total_seqs)
    
    #     num_samples = 0
    #     center = np.zeros(self.embed_size)

    #     if len(total_seqs) % batch_size == 0:
    #         num_batches = len(total_seqs) // batch_size
    #     else:
    #         num_batches = len(total_seqs) // batch_size + 1
        
    #     for i in range(num_batches):
    #         embeddings = self.model.predict(total_seqs[i*batch_size:(i+1)*batch_size])
    #         center += np.sum(embeddings, axis=0)
            
    #     hyper_center = center / len(total_seqs)
    #     return hyper_center
