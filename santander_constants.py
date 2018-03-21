from enum import Enum
import tensorflow as tf

PRODUCT_COLUMNS = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
                  'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
                  'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                  'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
                  'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
                  'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

MAX_SEQ_LENGTH = 18
N_INPUT = 48
N_OUTPUT = 24

class ModelType(Enum):
    RNN_NO_ATTENTION = 1
    RNN_ATTENTION_HS = 2
    RNN_ATTENTION_INPUT = 3
    RNN_ATTENTION_EMBEDDINGS = 4

class RnnCellType(Enum):
    LSTM = 1
    LSTM2 = 2
    GRU = 3
    RNN = 4
    LSTM_NORMALIZED = 5
    
class OptimizerType(Enum):
    SGD = 1
    ADAM = 2
    ADADELTA = 3
    ADAGRAD = 4

WEIGHT_INITIALIZATION_DICT = {
        'xavier' : lambda x: tf.contrib.layers.xavier_initializer(), # Xavier initialisation
        'normal' : lambda x: tf.random_normal_initializer(stddev=x), # Initialization from a standard normal
        'uniform': lambda x: tf.random_uniform_initializer(minval =-x, maxval=x) # Initialization from a uniform distribution
    }

class AttentionActivation(Enum):
    LINEAR = 1
    TANH = 2
    RELU = 3