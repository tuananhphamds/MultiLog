import tensorflow as tf
from official.nlp import optimization
from tensorflow.keras.callbacks import Callback
import wandb

def create_optimizer(num_X_train,
                     batch_size,
                     epochs):
    steps_per_epoch = num_X_train // batch_size
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)
    init_lr = 1e-4
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    return optimizer

def custom_loss(cce, vocab_size, ignore_index=1):
    def loss(y_true, y_pred):
        mask = (tf.math.argmax(y_true, axis=-1) != ignore_index)
        mask = tf.expand_dims(mask, axis=2)
        mask = tf.tile(mask, tf.constant([1, 1, vocab_size]))
        y_true_mask = y_true[mask]
        y_pred_mask = y_pred[mask]
        y_true_mask = tf.reshape(y_true_mask, [-1, vocab_size])
        y_pred_mask = tf.reshape(y_pred_mask, [-1, vocab_size])
        loss_output = cce(y_true_mask, y_pred_mask)
        return loss_output
    return loss

def custom_metric(acc_fn, vocab_size, ignore_index=1):
    def acc(y_true, y_pred):
        mask = (tf.math.argmax(y_true, axis=-1) != ignore_index)
        mask = tf.expand_dims(mask, axis=2)
        mask = tf.tile(mask, tf.constant([1, 1, vocab_size]))
        y_true_mask = y_true[mask]
        y_pred_mask = y_pred[mask]
        y_true_mask = tf.reshape(y_true_mask, [-1, vocab_size])
        y_pred_mask = tf.reshape(y_pred_mask, [-1, vocab_size])
        y_true_arg = tf.math.argmax(y_true_mask, axis=-1)
        y_pred_arg = tf.math.argmax(y_pred_mask, axis=-1)
        acc_fn.update_state(y_true=y_true_arg,
                         y_pred=y_pred_arg)
        accuracy = acc_fn.result()
        return accuracy
    return acc


def custom_contrast_loss(margin):
    def contrast_loss(y_true, y_pred):
        pos_loss = y_true * tf.math.pow(y_pred, 2)
        nev_loss = (1 - y_true) * tf.math.pow(tf.nn.relu(margin - y_pred), 2)
        loss = tf.math.reduce_mean(0.5 * (pos_loss + nev_loss))
        return loss
    return contrast_loss

def custom_distance(y_true, y_pred):
    distance = tf.linalg.norm(y_true - y_pred, axis=1)
    loss = tf.math.reduce_mean(distance)
    return loss


class ShowLRate(Callback):
    def on_epoch_end(self, epoch, logs=None):
        try:
            lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
        except:
            lr = self.model.optimizer.learning_rate
        #print('Learning rate at epoch {}: {}'.format(epoch, lr))

        # loss', 'mask_out_loss', 'next_out_loss', 'hypersphere_loss', 'mask_out_acc', 'next_out_acc', 'hypersphere_custom_distance'
        # print(logs['loss'])
        wandb.log({
            "loss": logs["loss"],
            "mask_out_loss": logs["mask_out_loss"],
            'next_out_loss': logs["next_out_loss"], 
            'hypersphere_loss': logs["hypersphere_loss"], 
            'mask_out_acc': logs["mask_out_acc"], 
            'next_out_acc': logs["next_out_acc"], 
            'hypersphere_custom_distance': logs["hypersphere_custom_distance"]
        })


    