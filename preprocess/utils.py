import os
import json
import numpy as np
HOME = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]

def check_dataset(dataset):
    if dataset not in ['bgl_drain',
                      'bgl_no_parser',
                      'bgl_spell',
                      'hdfs_drain',
                      'hdfs_no_parser',
                      'hdfs_spell',
                      'tbird_drain',
                      'tbird_no_parser',
                      'tbird_spell']:
        return False
    return True

def get_num_classes(dataset):
    template_path = os.path.join(HOME, 'datasets', dataset, 'templates.json')
    with open(template_path, 'r') as f:
        templates = json.load(f)
        return len(templates)
    raise Exception('Error in loading number of classes')

def load_template_embedding_matrix(num_spec_tokens,
                                   model_name,
                                   dataset,
                                   reduction_dim=None):
    log_template_dict = {
        'UNK': 0,
        'PAD': 1,
        'MASK': 2,
        'CLS': 3,
    }

    if reduction_dim is None:
        with open(os.path.join(HOME, 'datasets', dataset, '{}_embeddings.json'.format(model_name)), 'r') as f:
            embeddings = json.load(f)
    else:
        with open(os.path.join(HOME, 'datasets', dataset, '{}_embeddings_{}.json'.format(model_name, reduction_dim)), 'r') as f:
            embeddings = json.load(f)

    assert '1' in embeddings
    embed_size = len(embeddings['1'])

    for lkey, embed_vec in embeddings.items():
        log_template_dict[str(lkey)] = len(log_template_dict)

    embedding_matrix = np.zeros((len(log_template_dict), embed_size))
    for i in range(num_spec_tokens, len(log_template_dict)):
        embedding_matrix[i] = embeddings[str(i - num_spec_tokens + 1)]
    return log_template_dict, embedding_matrix

def generate_log_template_dict(vocab_size):
    log_template_dict = {
        'UNK': 0,
        'PAD': 1,
        'MASK': 2,
        'CLS': 3,
    }
    num_classes = vocab_size - len(log_template_dict)
    for i in range(num_classes):
        log_template_dict[str(i+1)] = len(log_template_dict)
    return log_template_dict