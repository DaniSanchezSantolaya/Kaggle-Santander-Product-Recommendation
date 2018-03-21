import santander_utils as utils
import santander_constants as constants
from dataset import DataSet
import pickle

# SPECIFY MODEL TYPE AND NAME, AND DATASET TO USE TO TRAIN
model_type = constants.ModelType.RNN_ATTENTION_EMBEDDINGS #'RNN_NO_ATTENTION'
name_model = 'rnn_attention_embeddings_include_new_customer_model1' #rnn_attention_hs_exp1'#rnn_no_attention_exp17'
path_dataset = "preprocessed/dataset_augmented_07_new_customer_feature.pickle"

# DEFINE PARAMETERS
parameters = {}
parameters['seq_length'] = constants.MAX_SEQ_LENGTH
parameters['n_input'] = constants.N_INPUT
parameters['n_output'] = constants.N_OUTPUT
parameters['n_hidden'] = 100
#parameters['init_stdev'] = 0.1
parameters['weight_init'] =  constants.WEIGHT_INITIALIZATION_DICT['normal'](0.5)
parameters['learning_rate'] = 0.001
parameters['optimizer'] = constants.OptimizerType.ADAM
parameters['rnn_type'] = constants.RnnCellType.LSTM2
parameters['num_epochs'] = 10000
parameters['batch_size'] = 128
parameters['embedding_size'] = 100              # Only for attention to embeddings model (will be ignored for others)
parameters['embedding_activation'] = 'linear'  # Only for attention to embeddings model (will be ignored for others)
parameters['attention_activation'] = constants.AttentionActivation.LINEAR
parameters['dropout'] = 0
parameters['l2'] = 0
parameters['layer_norm'] = False
print('RNN Type: ' + str(model_type))
print('Name: ' + str(name_model))
print(parameters)


# CREATE MODEL
model = utils.create_tensorflow_graph(model_type, parameters)

# LOAD DATASET
with open(path_dataset, 'rb') as handle:
    dataset = pickle.load(handle)
ds = DataSet(dataset)
del dataset

# TRAIN AND SAVE MODEL
utils.train_tf_model(model, ds, parameters, name_model)
