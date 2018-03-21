import santander_utils as utils
import santander_constants as constants
import pandas as pd
import pickle

# SPECIFY MODEL PATH, CHECKPOINT PATH AND SUBMISSION PATH
# Possible model types: rnn_no_attention, rnn_attention_hs, rnn_attention_input, rnn_attention_embeddings
model_type = constants.ModelType.RNN_ATTENTION_INPUT
checkpoint_path = 'C:/data/Santander/checkpoints/rnn_no_attention_FinalModel1/best_model/model_best.ckpt-2998144'
#checkpoint_path = './checkpoints/rnn_attention_embeddings_exp1/best_model/model_best.ckpt-2141568'
submission_path = 'submissions/rnn_no_attention_FinalModel1.csv'

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
parameters['batch_size'] = 1 #1 for generate predictoins
parameters['embedding_size'] = 100              # Only for attention to embeddings model (will be ignored for others)
parameters['embedding_activation'] = 'linear'  # Only for attention to embeddings model (will be ignored for others)
parameters['attention_activation'] = constants.AttentionActivation.LINEAR
parameters['dropout'] = 0.
parameters['l2'] = 0.
parameters['layer_norm'] = False
print('RNN Type: ' + str(model_type))
print(parameters)

# CREATE MODEL
model = utils.create_tensorflow_graph(model_type, parameters)

# LOAD TEST SET
df_test = pd.read_csv('test_ver2.csv')
with open("preprocessed/dataset_augmented_07.pickle", 'rb') as handle:
    dataset = pickle.load(handle)

# GENERATE PREDICTIONS
utils.compute_test_predictions(dataset, df_test, checkpoint_path, model, submission_path)
