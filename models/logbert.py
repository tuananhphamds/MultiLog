import os
import sys
import numpy as np
import random

HOME = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(HOME + '/preprocess')

from tqdm import tqdm
import tensorflow as tf
from base_model import BaseModel
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import Callback
from embeddinglayer_logbert import BERTEmbedding
from identity import Identity
from transformer import TransformerEncoder
from tensorflow.keras.callbacks import ModelCheckpoint
from dataloader import DataLoader
from subsequencegenerator import SubsequenceGenerator
from model_utils import ShowLRate
from model_utils import create_optimizer, custom_metric, custom_loss


def tf_nan(dtype):
    """Create NaN variable of proper dtype and variable shape for assign()."""
    return tf.Variable(float("nan"), dtype=dtype, shape=tf.TensorShape(None))

class HypercenterCallback(Callback):
    def __init__(self, hyper_center, num_data, embed_size):
        super(HypercenterCallback, self).__init__()
        self.y_predict = None
        self.hyper_center = hyper_center
        self.hyper_accumulate = np.zeros(embed_size)
        self.num_data = num_data
        self.embed_size = embed_size
        self.mse_loss = tf.keras.losses.MeanSquaredError()

    def set_model(self, model):
        self.model = model
        self.y_predict = tf_nan(model.output[0].dtype)

    def custom_distance(self, y_true, y_pred):
        self.y_predict.assign(y_pred)
        loss = self.mse_loss(self.hyper_center, y_pred)
        return loss

    def on_epoch_end(self, epoch, logs=None):
        self.hyper_accumulate = self.hyper_accumulate / self.num_data
        self.hyper_center = self.hyper_accumulate
        self.hyper_accumulate = np.zeros(self.embed_size)

    def on_train_batch_end(self, batch, logs=None):
        y_predict = self.y_predict.numpy()
        self.hyper_accumulate += np.sum(y_predict)


class LogBERT(BaseModel):
    def __init__(self, cfg, log_template_dict):
        super(LogBERT, self).__init__(cfg=cfg,
                                      log_template_dict=log_template_dict)
        self.hyper_center = None

    def _build_model(self):
        inputs = Input(shape=self.cfg['window_size'] + 2)
        embeddings = BERTEmbedding(vocab_size=self.cfg['vocab_size'],
                                   embed_size=self.cfg['embed_size'],
                                   window_size=self.cfg['window_size'] + 2)(inputs)
        trans_encoded = TransformerEncoder(num_blocks=self.cfg['num_blocks'],
                                           embed_size=self.cfg['embed_size'],
                                           num_heads=self.cfg['num_heads'],
                                           ff_dim=self.cfg['ff_dim'],
                                           window_size=self.cfg['window_size'] + 2,
                                           vocab_size=self.cfg['vocab_size'],
                                           dropout=self.cfg['dropout'])(embeddings)
        output1 = Dense(self.cfg['vocab_size'], activation='softmax', name='mask_out')(trans_encoded)
        output2 = Identity(name='hypersphere')(trans_encoded[:, 0])

        model = Model(inputs=inputs, outputs=[output1, output2])
        self.model = model
        print('Model summary', self.model.summary())

    def _load_model(self, model_file):
        print('----Loading model...')

        cce = tf.keras.losses.CategoricalCrossentropy()
        acc_fn = tf.keras.metrics.Accuracy()
        hyper_callback = HypercenterCallback(self.hyper_center, 10000, self.cfg['embed_size'])
        self.model = load_model(os.path.join(self.cfg['result_path'], model_file),
                                custom_objects={'TransformerEncoder': TransformerEncoder,
                                                              'AdamWeightDecay': create_optimizer(),
                                                              'BERTEmbedding': BERTEmbedding,
                                                              'loss': custom_loss(cce=cce, vocab_size=self.cfg['vocab_size'],
                                                                                  ignore_index=1),
                                                              'acc': custom_metric(acc_fn=acc_fn, vocab_size=self.cfg['vocab_size'],
                                                                                   ignore_index=1),
                                                              'Identity': Identity,
                                                              'custom_distance': hyper_callback.custom_distance})

    def _initialize_hyper_center(self, num_X_train, dataloader):
        hyper_center = np.zeros(self.cfg['embed_size'])

        for batch in dataloader:
            _, cls_encodings = self.model.predict(batch[0])
            hyper_center += np.sum(cls_encodings, axis=0)

        hyper_center = hyper_center / num_X_train
        return hyper_center

    def _compile(self, num_X_train):
        cce = tf.keras.losses.CategoricalCrossentropy()
        acc_fn = tf.keras.metrics.Accuracy()
        optimizer = create_optimizer(num_X_train=num_X_train,
                                     batch_size=self.cfg['batch_size'],
                                     epochs=self.cfg['epochs'])

        self.model.compile(loss={'mask_out': custom_loss(cce=cce, vocab_size=self.cfg['vocab_size'], ignore_index=1),
                                 'hypersphere': self.hyper_callback.custom_distance},
                           metrics={
                               'mask_out': custom_metric(acc_fn, vocab_size=self.cfg['vocab_size'], ignore_index=1),
                               'hypersphere': self.hyper_callback.custom_distance},
                           optimizer=optimizer,
                           loss_weights=self.cfg['loss_weights'])

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

        dataloader = DataLoader(X=X_train,
                                Y=Y_train,
                                batch_size=self.cfg['batch_size'],
                                vocab_size=self.cfg['vocab_size'],
                                log_template_dict=self.log_template_dict,
                                shuffle=True,
                                window_size=self.cfg['window_size'],
                                hyper_center=self.hyper_center,
                                mask_rate=self.cfg['mask_rate'],
                                model_name=self.cfg['model_name'])

        self.hyper_center = self._initialize_hyper_center(num_X_train=len(X_train),
                                                          dataloader=dataloader)

        self.hyper_callback = HypercenterCallback(self.hyper_center, len(X_train), self.cfg['embed_size'])

        self._compile(len(X_train))
        history = self.model.fit(dataloader,
                                 epochs=self.cfg['epochs'],
                                 callbacks=[model_checkpoint, ShowLRate(), self.hyper_callback],
                                 verbose=self.cfg['verbose'])

        self._save_train_results_as_plot(history.history)


    def _random_template(self, lk_seq):
        output_label = []
        output = []

        # 15% of the tokens would be replaced
        for i, lk in enumerate(lk_seq):
            prob = random.random()
            if lk != 0:
                lk_id = self.log_template_dict[str(lk)]
            else:
                lk_id = self.log_template_dict['UNK']

            if prob < self.cfg['mask_rate']:
                prob /= self.cfg['mask_rate']

                # 80% chance change token to mask token
                if prob < 0.8:
                    output.append(self.log_template_dict['MASK'])

                # 10% chance change token to random token
                elif prob < 0.9:
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

    def _calculate_scores_all_sessions(self, sess_dct, generator_batch_size):
        sub_generator = SubsequenceGenerator(sess_dct,
                                             batch_size=generator_batch_size,
                                             window_size=self.cfg['window_size'])

        total_scores = []

        for batch_data in tqdm(sub_generator.get_batches()):
            subseqs = []
            next_labels = []

            for i in range(len(batch_data)):
                subseqs += batch_data[i][0]
                next_labels += batch_data[i][1]

            mask_subseqs = []
            mask_labels = []
            for subseq, next_label in zip(subseqs, next_labels):
                mask_subseq, mask_label = self._random_template(subseq + (next_label,))
                mask_subseqs.append([self.log_template_dict['CLS']] + mask_subseq)
                mask_labels.append([1] + mask_label)

            mask_subseqs = np.array(mask_subseqs).reshape(len(subseqs), self.cfg['window_size'] + 2)
            mask_labels = np.array(mask_labels).reshape(len(subseqs), self.cfg['window_size'] + 2, 1)

            mask_output, _ = self.model.predict(mask_subseqs)
            argsorted = np.argsort(mask_output)
            candidates = np.where(argsorted == mask_labels)[-1]
            candidates = candidates.reshape(-1, self.cfg['window_size'] + 2)
            mask_labels1 = (mask_labels.reshape(len(subseqs), self.cfg['window_size'] + 2) != 1).astype(float)
            candidates = candidates * mask_labels1
            mask_labels2 = (mask_labels.reshape(len(subseqs), self.cfg['window_size'] + 2) == 1).astype(float)
            mask_labels2 = mask_labels2 * self.cfg['vocab_size']
            candidates = candidates + mask_labels2
            candidates = np.sort(candidates, axis=-1)

            total_scores.append(candidates)

        return total_scores

    def test(self, normal_sess_dct, abnormal_sess_dct, X_val=None, Y_val=None):
        self._load_model(self.cfg['model_file'])

        # Calculate scores per session
        print('Testing normal data')
        all_normal_probs = self._calculate_scores_all_sessions(normal_sess_dct, self.cfg['generator_batch_size'])
        print('Testing abnormal data')
        all_abnormal_probs = self._calculate_scores_all_sessions(abnormal_sess_dct, self.cfg['generator_batch_size'])

        normal_prob_per_session, normal_freqs = self._calculate_scores_per_session(normal_sess_dct,
                                                                                   self.cfg['generator_batch_size'],
                                                                                   all_normal_probs,
                                                                                   top_k=self.cfg['top_k'])
        abnormal_probs_per_session, abnormal_freqs = self._calculate_scores_per_session(abnormal_sess_dct,
                                                                                        self.cfg['generator_batch_size'],
                                                                                        all_abnormal_probs,
                                                                                        top_k=self.cfg['top_k'])

        probs_results = self._get_best_results(normal_prob_per_session,
                                               abnormal_probs_per_session,
                                               normal_freqs,
                                               abnormal_freqs)


        print(probs_results)

        self._save_best_results(probs_results)
        
