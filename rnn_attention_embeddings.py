import tensorflow as tf
import santander_utils as utils
import santander_constants as constants
import sys



# DEFINE MODEL
class RNNAttentionEmbeddings:

    def __init__(self, parameters):
        self.parameters = parameters

        tf.reset_default_graph()
        # Define placeholders
        self.x = tf.placeholder("float", [None, parameters['seq_length'], parameters['n_input']], name='x')
        self.y = tf.placeholder("float", [None, parameters['n_output']], name='y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Define weights and bias - For now we will try with attention to hidden state 
        #self.weights = {
        #    'alphas': tf.Variable(parameters['weight_init']([parameters['n_hidden'], 1]), name='w_alphas'),
        #    'out': tf.Variable(parameters['weight_init']([parameters['embedding_size'], parameters['n_output']]), name='w_out'),
        #    'emb': tf.Variable(parameters['weight_init']([parameters['n_input'], parameters['embedding_size']]), name='w_emb')
        #}
        print(parameters['weight_init'])
        self.weights = {
            'alphas': tf.get_variable(name='w_alphas', shape=[parameters['n_hidden'], 1], initializer=parameters['weight_init']),
            'out': tf.get_variable(name='w_out', shape=[parameters['embedding_size'], parameters['n_output']], initializer=parameters['weight_init']),
            'emb': tf.get_variable(name='w_emb', shape=[parameters['n_input'], parameters['embedding_size']], initializer=parameters['weight_init'])
        }


        self.biases = {
            'out': tf.Variable(tf.zeros([parameters['n_output']]), name='b_out'),
            'alphas': tf.Variable(tf.zeros([1]), name='b_alphas'),
            'emb': tf.Variable(tf.zeros([parameters['embedding_size']]), name='b_emb')
        }

        # Compute embeddings
        self.x_reshaped = tf.reshape(self.x, [-1, int(self.x.get_shape()[2])])
        if parameters['embedding_activation'] == 'linear':
            self.v = tf.matmul(self.x_reshaped, self.weights['emb'])
        elif parameters['embedding_activation'] == 'tanh':
            self.v = tf.tanh(tf.matmul(self.x_reshaped, self.weights['emb']) + self.biases['emb'])
        elif parameters['embedding_activation'] == 'sigmoid':
            self.v = tf.sigmoid(tf.matmul(self.x_reshaped, self.weights['emb']) + self.biases['emb'])
        self.v_reshaped = tf.reshape(self.v, [-1, parameters['seq_length'], parameters['embedding_size']])
        if parameters['layer_norm']:
            self.v_reshaped = tf.contrib.layers.layer_norm(self.v_reshaped)

        # Define RNN
        rnn_cell = utils.get_rnn_cell(parameters)
        #Add dropout
        if parameters['dropout'] > 0:
            rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=self.dropout_keep_prob)
        self.seq_lenghts = utils.seq_length(self.v_reshaped)
        self.outputs, states = tf.nn.dynamic_rnn(
            rnn_cell,
            self.v_reshaped,
            dtype=tf.float32,
            sequence_length=self.seq_lenghts
        )


        # Define attention weihts
        self.outputs_reshaped = tf.reshape(self.outputs, [-1, int(self.outputs.get_shape()[2])])
        if parameters['attention_activation'] == constants.AttentionActivation.LINEAR:
            self.ejs = tf.matmul(self.outputs_reshaped, self.weights['alphas']) + self.biases['alphas']
        elif parameters['attention_activation'] == constants.AttentionActivation.RELU:
            self.ejs = tf.nn.relu(tf.matmul(self.outputs_reshaped, self.weights['alphas']) + self.biases['alphas'])
        elif parameters['attention_activation'] == constants.AttentionActivation.TANH:
            self.ejs = tf.nn.tanh(tf.matmul(self.outputs_reshaped, self.weights['alphas']) + self.biases['alphas'])
        else:
            print('Specified attention activation not valid')
            sys.exit()
        self.ejs_reshaped = tf.reshape(self.ejs, [-1, int(self.outputs.get_shape()[1])])
        self.alphas_all = tf.nn.softmax(self.ejs_reshaped, name='attention_weights')
        self.seq_mask = tf.sequence_mask(self.seq_lenghts, maxlen=parameters['seq_length'])
        self.alphas_masked = tf.multiply(self.alphas_all, tf.cast(self.seq_mask, tf.float32))
        self.sum_alphas = tf.reduce_sum(self.alphas_masked, axis=1)
        self.reshape_sum_alphas = tf.reshape(self.sum_alphas, [-1, 1])
        self.alphas = tf.div(self.alphas_masked, self.reshape_sum_alphas)
        #self.alphas = tf.div(self.alphas_masked, tf.reshape(self.sum_alphas, [parameters['batch_size'],1]))
        #self.alphas = tf.nn.log_softmax(self.ejs_reshaped, name='attention_weights')
        self.reshaped_alphas = tf.reshape(self.alphas, [-1, 1])
        # Define context
        self.context = self.reshaped_alphas * self.v
        self.context_reshaped = tf.reshape(self.context, [-1, parameters['seq_length'], int(self.context.get_shape()[1])])
        self.context_reduced = tf.reduce_sum(self.context_reshaped, axis= 1)

        # Normalize context by number of timesteps?
        # Define logits and loss
        self.logits = tf.matmul(self.context_reduced, self.weights['out']) + self.biases['out']
        self.pred_prob = tf.sigmoid(self.logits, name="predictions")
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        # L2 regularization
        for var in tf.trainable_variables():
            if ('b_out' not in var.name) and ('b_alphas' not in var.name) and ('b_emb' not in var.name) and ('bias' not in var.name) and ('LayerNorm' not in var.name):
                print('Variable ' + var.name + ' will be regularized')
                self.loss += parameters['l2'] * tf.nn.l2_loss(var)
            

        #Define optimizer
        self.optimizer = utils.get_optimizer(parameters, self.loss)

        # Initialization
        self.init = tf.global_variables_initializer()

        #Add summaries
        utils.add_all_summaries(self.loss, include_gradients=True)