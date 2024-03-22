import os
import sys
import re
import json

HOME = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(HOME + '/preprocess')
sys.path.append(HOME + '/models')

from optparse import OptionParser
from datetime import datetime
from datagenerator import DataGenerator
from n_bert import NBert
from pn_bert import PNBert
from multilog import PNHBert
from logbert import LogBERT
from deeplog import DeepLog
from loganomaly import LogAnomaly
from utils import get_num_classes, load_template_embedding_matrix, generate_log_template_dict
from model_config import logbert_config, deeplog_config, loganomaly_config, n_bert_config, pn_bert_config, multilog_config
from centersphere import CenterSphere


class Trainer:
    def __init__(self, cfg):
        self.result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
        self.cfg = cfg

        # Create a result directory
        self.cfg['result_path'] = self._create_result_path(self.result_dir)

        # Save model config
        self._save_model_config()

    def _save_model_config(self):
        print('--------------------SAVING MODEL CONFIG--------------------')
        with open(os.path.join(self.cfg['result_path'], 'config.json'), 'w') as f:
            json.dump(self.cfg, f, indent=4)

    def _create_result_path(self, result_dir):
        now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        result_path = now
        key_list = ['model_name', 'pretrained_model', 'train_val_test', 'shuffle', 'random_seed', 'num_blocks',
                    'num_heads', 'dataset']
        for key in key_list:
            if key in self.cfg:
                result_path += '_' + re.sub('[\s\(\),]', '_', str(self.cfg[key]))

        result_abs_path = os.path.join(result_dir, result_path)
        if not os.path.exists(result_abs_path):
            os.mkdir(result_abs_path)
        return result_abs_path

    def run(self):
        data_generator = DataGenerator(dataset=self.cfg['dataset'],
                                       window_size=self.cfg['window_size'],
                                       train_val_test=self.cfg['train_val_test'],
                                       val_pos=self.cfg['val_pos'],
                                       shuffle=self.cfg['shuffle'],
                                       random_seed=self.cfg['random_seed'],
                                       fixed_window=self.cfg['fixed_window'])
        split_data = data_generator.generate_train_val_test()
        X_train, Y_train = split_data['train']
        X_val, Y_val = split_data['val']

        normal_sess_dct, abnormal_sess_dct = split_data['test']

        # load log template dict, embedding matrix
        if self.cfg['model_name'] not in ['deeplog', 'logbert']:
            log_template_dict, embedding_matrix = load_template_embedding_matrix(
                num_spec_tokens=self.cfg['num_spec_tokens'],
                model_name=self.cfg['pretrained_model'],
                dataset=self.cfg['dataset'],
                reduction_dim=self.cfg['reduction_dim'])
        else:
            log_template_dict = generate_log_template_dict(self.cfg['vocab_size'])

        # Build model
        if self.cfg['model_name'] == 'n_bert':
            model = NBert(self.cfg, embedding_matrix, log_template_dict)
        elif self.cfg['model_name'] == 'pn_bert':
            model = PNBert(self.cfg, embedding_matrix, log_template_dict)
        elif self.cfg['model_name'] == 'multilog':
            csphere = CenterSphere(embedding_matrix=embedding_matrix,
                                   num_vocab=self.cfg['vocab_size'],
                                   window_size=self.cfg['window_size'],
                                   embed_size=self.cfg['reduction_dim'],
                                   num_spec_tokens=self.cfg['num_spec_tokens'],
                                   log_template_dict=log_template_dict)
            hyper_center = csphere.calculate_center_sphere(X_train).reshape(1, -1)
            model = PNHBert(self.cfg, embedding_matrix, log_template_dict, hyper_center)
        elif self.cfg['model_name'] == 'logbert':
            model = LogBERT(self.cfg, log_template_dict)
        elif self.cfg['model_name'] == 'deeplog':
            model = DeepLog(self.cfg, log_template_dict)
        elif self.cfg['model_name'] == 'loganomaly':
            model = LogAnomaly(self.cfg, embedding_matrix, log_template_dict)

        model.train(X_train=X_train,
                    Y_train=Y_train,
                    X_val=X_val,
                    Y_val=Y_val)

        model.test(normal_sess_dct, abnormal_sess_dct, X_val=X_val, Y_val=Y_val)

def parse_options():
    parser = OptionParser(usage='Training a model')
    parser.add_option('-d', '--dataset', action='store', type='str', dest='dataset', default='hdfs_drain')
    parser.add_option('-m', '--model', action='store', type='str', dest='model', default='multilog')
    parser.add_option('-t', '--train_rate', action='store', type='int', dest='train_rate', default=1)
    parser.add_option('-e', '--test_rate', action='store', type='int', dest='test_rate', default=20)
    parser.add_option('-r', '--random_seed', action='store', type='int', dest='random_seed', default=50)
    parser.add_option('-p', '--epochs', action='store', type='int', dest='epochs', default=50)
    parser.add_option('-c', '--use_cfg_file', action='store', dest='use_cfg_file', default="True")

    (options, args) = parser.parse_args()
    return options

if __name__ == "__main__":
    config = {
        'dataset': 'hdfs_drain',
        'model_name': 'deeplog',
        'train_val_test': (80, 0, 20),
        'num_spec_tokens': 4,
    }

    config_mapping = {
        'logbert': logbert_config,
        'deeplog': deeplog_config,
        'loganomaly': loganomaly_config,
        'n_bert': n_bert_config,
        'pn_bert': pn_bert_config,
        'multilog': multilog_config
    }

    options = parse_options()
    config['dataset'] = options.dataset
    config['model_name'] = options.model

    if config['model_name'] not in config_mapping:
        raise ValueError('{} is not supported. Only support {}'.format(config['model_name'],
                                                                       list(config_mapping.keys())))

    config.update(config_mapping[config['model_name']])
    config.update({
        'vocab_size': get_num_classes(config['dataset']) + config['num_spec_tokens']
    })

    trainer = Trainer(config)
    trainer.run()



