import tensorflow as tf
import santander_utils as utils



# DEFINE MODEL
class Rnn:

    def __init__(self, parameters):
        self.parameters = parameters
    
        tf.reset_default_graph()

        # Define placeholders
        self.x = tf.placeholder("float", [None, parameters['seq_length'], parameters['n_input']], name='x')
        self.y = tf.placeholder("float", [None, parameters['n_output']], name='y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Define weights and bias - For now we will try with attention to hidden state 
        self.weights = {
            'out': tf.get_variable(name='w_out', shape=[parameters['n_hidden'], parameters['n_output']], initializer=parameters['weight_init'])
        }

        self.biases = {
            'out': tf.Variable(tf.zeros([parameters['n_output']]), name='b_out')
        }


        # Define RNN
        rnn_cell = utils.get_rnn_cell(parameters)
        #Add dropout
        if parameters['dropout'] > 0:
            rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=self.dropout_keep_prob)
        self.outputs, states = tf.nn.dynamic_rnn(
            rnn_cell,
            self.x,
            dtype=tf.float32,
            sequence_length=utils.seq_length(self.x)
        )
        self.last_relevant_output = utils.last_relevant(self.outputs, utils.seq_length(self.x))

        # Define logits and loss
        self.final_logits = tf.matmul(self.last_relevant_output, self.weights['out']) + self.biases['out']
        self.pred_prob = tf.sigmoid(self.final_logits, name="predictions")
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.final_logits, labels=self.y))
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