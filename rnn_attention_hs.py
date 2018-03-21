import numpy as np
import tensorflow as tf
import santander_utils as utils



# DEFINE MODEL
class RnnAttentionHS:

    def __init__(self, parameters):
        self.parameters = parameters

        tf.reset_default_graph()

        # Define placeholders
        self.x = tf.placeholder("float", [None, parameters['seq_length'], parameters['n_input']], name='x')
        self.y = tf.placeholder("float", [None, parameters['n_output']], name='y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Define weights and bias - For now we will try with attention to hidden state
        self.weights = {
            'alphas': tf.get_variable(name='w_alphas', shape=[parameters['n_hidden'], 1], initializer=parameters['weight_init']),
            'out': tf.get_variable(name='w_out', shape=[parameters['n_hidden'], parameters['n_output']], initializer=parameters['weight_init'])
        }

        self.biases = {
            'out': tf.Variable(tf.random_normal([parameters['n_output']]), name='b_out'),
            'alphas': tf.Variable(tf.random_normal([1]), name='b_alphas')
        }

        # Define RNN
        rnn_cell = utils.get_rnn_cell(parameters)
        # Add dropout
        if parameters['dropout'] > 0:
            rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=self.dropout_keep_prob)
        self.seq_lenghts = utils.seq_length(self.v_reshaped)
        self.outputs, states = tf.nn.dynamic_rnn(
            rnn_cell,
            self.x,
            dtype=tf.float32,
            sequence_length=self.seq_lenghts
        )

        # Define attention weihts
        self.outputs_reshaped = tf.reshape(self.outputs, [-1, int(self.outputs.get_shape()[2])])
        self.ejs = tf.matmul(self.outputs_reshaped, self.weights['alphas'])  + self.biases['alphas']
        self.ejs_reshaped = tf.reshape(self.ejs, [-1, int(self.outputs.get_shape()[1])])        
        self.alphas_all = tf.nn.softmax(self.ejs_reshaped, name='attention_weights')
        self.seq_mask = tf.sequence_mask(self.seq_lenghts, maxlen=parameters['seq_length'])
        self.alphas_masked = tf.multiply(self.alphas_all, tf.cast(self.seq_mask, tf.float32))
        self.sum_alphas = tf.reduce_sum(self.alphas_masked, axis=1)
        self.alphas = tf.div(self.alphas_masked, tf.reshape(self.sum_alphas, [parameters['batch_size'],1]))
        self.reshaped_alphas = tf.reshape(self.alphas, [-1, 1])
        # Define context
        self.context = self.reshaped_alphas * self.outputs_reshaped
        self.context_reshaped = tf.reshape(self.context, [-1, parameters['seq_length'], int(self.context.get_shape()[1])])
        self.context_reduced = tf.reduce_sum(self.context_reshaped, axis=1)

        # Normalize context by number of timesteps?
        # Define logits and loss
        self.logits = tf.matmul(self.context_reduced, self.weights['out']) + self.biases['out']
        self.pred_prob = tf.sigmoid(self.logits, name="predictions")
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        # L2 regularization
        for var in tf.trainable_variables():
            if ('b_out' not in var.name) and ('b_alphas' not in var.name) and ('b_emb' not in var.name) and (
                    'bias' not in var.name) and ('LayerNorm' not in var.name):
                print('Variable ' + var.name + ' will be regularized')
                self.loss += parameters['l2'] * tf.nn.l2_loss(var)

        # Define optimizer
        self.optimizer = utils.get_optimizer(parameters, self.loss)

        # Initialization
        self.init = tf.global_variables_initializer()

        # Add summaries
        utils.add_all_summaries(self.loss, include_gradients = True)