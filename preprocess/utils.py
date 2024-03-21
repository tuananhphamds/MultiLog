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

def get_feature_size(model_name):
    """
    model_name:
    - all_base:             768
    - all_distill:          768
    - all_mini_6:           384
    - all_mini_12:          384
    - bert_base:            768
    - bert_mini:            256
    - multi_qa:             384
    - multilingual_base:    768
    - multilingual_mini:    384
    """
    if model_name in ['all_base', 'all_distill', 'bert_base', 'multilingual_base']:
        return 768
    elif model_name in ['all_mini_6', 'all_mini_12', 'multi_qa', 'multilingual_mini']:
        return 384
    elif model_name in ['bert_mini']:
        return 256
    else:
        raise ValueError(f'Model name {model_name} is not supported')

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