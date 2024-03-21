import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import sys
import re
import json
import numpy as np
import wandb
from multiprocessing import Process

HOME = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(HOME + '/preprocess')
sys.path.append(HOME + '/models')

from datagenerator import DataGenerator
from n_bert import NBert
from pn_bert import PNBert
from pnh_bert import PNHBert
from pnh_bert_euclidean import PNHBertEuclidean
from pnl_bert_euclidean import PNLBertEuclidean
from pnl_bert_euclidean_average import PNLBertEuclideanAverage
from utils import get_num_classes, get_feature_size, load_template_embedding_matrix
from centersphere import CenterSphere
from datetime import datetime


class Trainer:
    def __init__(self, cfg):
        self.result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
        self.cfg = cfg

        # Create a result directory
        self.cfg['result_path'] = self._create_result_path(self.result_dir)
        #self.cfg['result_path'] = os.path.join(self.result_dir, '2024_03_19_09_30_31_pnh_bert_multilingual_mini__80__0__20__True_50_2_6_tbird_no_parser/')

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
        log_template_dict, embedding_matrix = load_template_embedding_matrix(num_spec_tokens=self.cfg['num_spec_tokens'],
                                                                             model_name=self.cfg['pretrained_model'],
                                                                             dataset=self.cfg['dataset'],
                                                                             reduction_dim=self.cfg['reduction_dim'])

        # Build model
        if self.cfg['model_name'] == 'n_bert':
            model = NBert(self.cfg, embedding_matrix, log_template_dict)
        elif self.cfg['model_name'] == 'pn_bert':
            model = PNBert(self.cfg, embedding_matrix, log_template_dict)
        elif self.cfg['model_name'] == 'pnh_bert':
            csphere = CenterSphere(embedding_matrix=embedding_matrix,
                                   num_vocab=self.cfg['vocab_size'],
                                   window_size=self.cfg['window_size'],
                                   embed_size=self.cfg['embed_size'],
                                   num_spec_tokens=self.cfg['num_spec_tokens'],
                                   log_template_dict=log_template_dict)
            hyper_center = csphere.calculate_center_sphere(X_train).reshape(1, -1)
            model = PNHBert(self.cfg, embedding_matrix, log_template_dict, hyper_center)
        elif self.cfg['model_name'] == 'pnh_bert_euclidean':
            new_train = []
            for x, y in zip(X_train, Y_train):
                new_train.append(list(x) + [y])
            csphere = CenterSphere(embedding_matrix=embedding_matrix,
                                   num_vocab=self.cfg['vocab_size'],
                                   window_size=self.cfg['window_size'] + 1,
                                   embed_size=self.cfg['embed_size'],
                                   num_spec_tokens=self.cfg['num_spec_tokens'],
                                   log_template_dict=log_template_dict)
            hyper_center = csphere.calculate_center_sphere(new_train)
            model = PNHBertEuclidean(self.cfg, embedding_matrix, log_template_dict, hyper_center)
        elif self.cfg['model_name'] == 'pnl_bert_euclidean':
            model = PNLBertEuclidean(self.cfg, embedding_matrix, log_template_dict)
        elif self.cfg['model_name'] == 'pnl_bert_euclidean_average':
            model = PNLBertEuclideanAverage(self.cfg, embedding_matrix, log_template_dict)

        model.train(X_train=X_train,
                 Y_train=Y_train,
                 X_val=X_val,
                 Y_val=Y_val)

        model.test(normal_sess_dct, abnormal_sess_dct, X_val=X_val, Y_val=Y_val)
        #model.test_insert(normal_sess_dct, abnormal_sess_dct, X_val=X_val, Y_val=Y_val)

def run(config, option_name, project):
    wandb.init(
                project=project,
                name=option_name,
                config=config
            ) 
    try:
        trainer = Trainer(config)
        trainer.run()
    except Exception as e:
        with open('errors.log', 'a', encoding='utf8') as f:
            f.write('\n\n\nFailed to train model with option {} {}'.format(option_name, str(e)))
            print('\n\n\nFailed to train model with option {} {}'.format(option_name, str(e)))
    
    wandb.finish()


if __name__ == "__main__":
    config = {
        'dataset': 'hdfs_no_parser',
        'pretrained_model': 'multilingual_mini',
        'model_name': 'pn_bert',
        'window_size': 10,
        'fixed_window': 20,
        'num_spec_tokens': 4,
        'train_val_test': (80, 0, 20),
        'val_pos': 'head',
        'shuffle': True,
        'random_seed': 50,
        'top_k': 1,
        'mask_rate': 0.15,
        'embed_size': 32,
        'reduction_dim': 32
    }

    num_classes = get_num_classes(config['dataset'])
    config['vocab_size'] = num_classes + config['num_spec_tokens']
    #config['embed_size'] = get_feature_size(config['pretrained_model'])

    model_config = {
        'model_file': 'best_model.hdf5',
        'batch_size': 512,
        'batch_size_val': 64,
        'dropout': 0.1,
        'num_heads': 6,
        'num_blocks': 2,
        'ff_dim': 1024,
        'epochs': 50,
        'verbose': 1
    }

    if config['model_name'] == 'pn_bert':
        model_config['loss_weights'] = {
            'mask_out': 0.5,
            'next_out': 0.5
        }
    elif config['model_name'] in ['pnh_bert']:
        model_config['loss_weights'] = {
            'mask_out': 0.5,
            'next_out': 0.4,
            'hypersphere': 0.1
        }
        config['top_k'] = 1
    elif config['model_name'] in ['pnh_bert_euclidean', 'pnl_bert_euclidean', 'pnl_bert_euclidean_average']:
        model_config['loss_weights'] = {
            'mask_out': 1,
            'next_out': 1,
            'contrast_out': 0.1
        }
        config['top_k'] = 1

    config.update(model_config)

    #trainer = Trainer(config)
    #trainer.run()

    option_name = 'dataset'
    project = 'training_objectives'
    options1 = ['pn_bert']
    options = ['hdfs_no_parser', 
               'bgl_no_parser', 
               'tbird_no_parser']
    for option1 in options1:
        for option in options:
            config[option_name] = option
            config['model_name'] = option1
            num_classes = get_num_classes(option)
            config['vocab_size'] = num_classes + config['num_spec_tokens']
            
            p = Process(target=run, args=(config, '{}_{}'.format(option_name, option), project))
            p.start()
            p.join()
        
    
    
