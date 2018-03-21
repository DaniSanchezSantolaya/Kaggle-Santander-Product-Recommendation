import tensorflow as tf
import numpy as np
import santander_constants as constants
from rnn_no_attention import Rnn
from rnn_attention_hs import RnnAttentionHS
from rnn_attention_input import RNNAttentionInput
from rnn_attention_embeddings import RNNAttentionEmbeddings
import sys
import pandas as pd
import pickle
import os

#####################################################
# Some auxilar functions to create TensorFlow models
#####################################################
def seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length
def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)

    return relevant

def get_rnn_cell(parameters):
    """ Function to obtain rnn cell
    """
    if parameters['rnn_type'] == constants.RnnCellType.LSTM:
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(parameters['n_hidden'], forget_bias=1.0)
    elif parameters['rnn_type'] == constants.RnnCellType.LSTM2:
        rnn_cell = tf.contrib.rnn.LSTMCell(parameters['n_hidden'])
    elif parameters['rnn_type'] == constants.RnnCellType.GRU:
        rnn_cell = tf.contrib.rnn.GRUCell(parameters['n_hidden'])
    elif parameters['rnn_type'] == constants.RnnCellType.RNN:
        rnn_cell = tf.contrib.rnn.BasicRNNCell(parameters['n_hidden'])
    elif parameters['rnn_type'] == constants.RnnCellType.LSTM_NORMALIZED:
        rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(parameters['n_hidden'])
    return rnn_cell

def get_optimizer(parameters, loss):
    """ Function to obtain optimizer
    """
    if parameters['optimizer'] == constants.OptimizerType.SGD:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=parameters['learning_rate']).minimize(loss)
    elif parameters['optimizer'] == constants.OptimizerType.ADAM:
        optimizer = tf.train.AdamOptimizer(learning_rate=parameters['learning_rate']).minimize(loss)
    elif parameters['optimizer'] == constants.OptimizerType.ADADELTA:
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=parameters['learning_rate']).minimize(loss)
    elif parameters['optimizer'] == constants.OptimizerType.ADAGRAD:
        optimizer = tf.train.AdagradOptimizer(learning_rate=parameters['learning_rate']).minimize(loss)
    return optimizer


def create_tensorflow_graph(model_type, parameters):
    if model_type == constants.ModelType.RNN_NO_ATTENTION:
        model = Rnn(parameters)
    elif model_type == constants.ModelType.RNN_ATTENTION_HS:
        model = RnnAttentionHS(parameters)
    elif model_type == constants.ModelType.RNN_ATTENTION_INPUT:
        model = RNNAttentionInput(parameters)
    elif model_type == constants.ModelType.RNN_ATTENTION_EMBEDDINGS:
        model = RNNAttentionEmbeddings(parameters)
    else:
        print('Model type not valid')
        sys.exit()
    return model


def add_all_summaries(loss, include_gradients = True):
    """
    Add summaries to monitor in tensorboard
    :param loss: loss to be monitored
    :param include_gradients:  if we want to monitor
    :return:
    """
    tf.summary.scalar('loss', loss)
    # Create summaries to visualize weights
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    # Summarize all gradients
    grads = tf.gradients(loss, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.name + '/gradient', grad)

#####################################################
# FUNCTION TO TRAIN TENSORFLOW MODEL
#####################################################
def train_tf_model(model, ds, parameters, name_model):
    """ Function to train a model
    """
    
    path_export_model = "protobuf_models/" + name_model + "/"
    display_train_loss = 200
    steps_periodic_checkpoint = 200
    current_epoch = 0

    # Start training
    saver_last = tf.train.Saver()
    saver_best = tf.train.Saver()
    checkpoint_dir = './checkpoints/' + name_model + '/'
    if not tf.gfile.Exists(checkpoint_dir):
        tf.gfile.MakeDirs(checkpoint_dir)
        tf.gfile.MakeDirs(checkpoint_dir + '/best_model')
        tf.gfile.MakeDirs(checkpoint_dir + '/last_model')
    best_loss = 150000000
    with tf.Session() as sess:

        # Create FileWriters for summaries
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('tensorboard/' + name_model + '/train', sess.graph)
        val_writer = tf.summary.FileWriter('tensorboard/' + name_model + '/val', sess.graph)
        
        # Run the initializer
        sess.run(model.init)

        step = 1
        while ds.get_current_epoch('train') < parameters['num_epochs']:
            
            # Get next batch
            batch_x, batch_y = ds.next_batch(parameters['batch_size'])
            
            # Run optimization op (backprop)
            _ = sess.run(model.optimizer, feed_dict={model.x: batch_x, model.y: batch_y, model.dropout_keep_prob: (1 - parameters['dropout'])})
            
            # Compute train loss
            if step % display_train_loss == 0 or step == 1:
                # Calculate batch loss
                train_loss, summary = sess.run([model.loss, merged], feed_dict={model.x: batch_x, model.y: batch_y, model.dropout_keep_prob: 1})
                print("Step " + str(step) + ", Train Loss: " + str(train_loss))
                train_writer.add_summary(summary, step * parameters['batch_size'])
                sys.stdout.flush()
                
            
            # Periodic model checkpoint
            if step % steps_periodic_checkpoint == 0:
                checkpoint_dir_tmp = checkpoint_dir + '/last_model/'
                checkpoint_path = os.path.join(checkpoint_dir_tmp, 'model_last.ckpt')
                saver_last.save(sess, checkpoint_path, global_step=step*parameters['batch_size'])
                
            # Compute val loss and save model at the end of each epoch
            if ds.get_current_epoch('train') != current_epoch:
                current_epoch = ds.get_current_epoch('train')
                # Train error
                X_train, Y_train = ds.get_set('train')
                start_idx = 0
                end_idx = start_idx + parameters['batch_size']
                train_losses = []
                while end_idx < len(X_train):
                    batch_x = X_train[start_idx:end_idx]
                    batch_y = Y_train[start_idx:end_idx]
                    train_loss, summary = sess.run([model.loss, merged], feed_dict={model.x: batch_x, model.y: batch_y, model.dropout_keep_prob: 1})
                    train_losses.append(train_loss)
                    start_idx = end_idx
                    end_idx = start_idx + parameters['batch_size']
                train_loss = np.mean(train_losses)
                print("----End epoch " + str(current_epoch - 1) + ", Train Loss: " + str(train_loss))
                #train_writer.add_summary(summary, step * parameters['batch_size'])
                # Validation error
                X_val, Y_val = ds.get_set('val')
                start_idx = 0
                end_idx = start_idx + parameters['batch_size']
                val_losses = []
                while end_idx < len(X_val):
                    batch_x = X_val[start_idx:end_idx]
                    batch_y = Y_val[start_idx:end_idx]
                    val_loss, summary = sess.run([model.loss, merged], feed_dict={model.x: batch_x, model.y: batch_y, model.dropout_keep_prob: 1})
                    val_losses.append(val_loss)
                    start_idx = end_idx
                    end_idx = start_idx + parameters['batch_size']
                val_loss = np.mean(val_losses)
                print("----End epoch " + str(current_epoch - 1) + ", Val Loss: " + str(val_loss))
                print('--------------------------------------------------------')
                val_writer.add_summary(summary, step * parameters['batch_size'])
                sys.stdout.flush()
                
                # Check if validation loss is better
                if val_loss < best_loss:
                    best_loss = val_loss
                    checkpoint_dir_tmp = checkpoint_dir + '/best_model/'
                    checkpoint_path = os.path.join(checkpoint_dir_tmp, 'model_best.ckpt')
                    saver_best.save(sess, checkpoint_path, global_step=step*parameters['batch_size'])

                
                # Saved Model Builder 
                export_path = path_export_model + "epoch" + str(current_epoch - 1)
                builder = tf.saved_model.builder.SavedModelBuilder(export_path)
                builder.add_meta_graph_and_variables(
                      sess, [tf.saved_model.tag_constants.SERVING])
                builder.save()
                
            step += 1

        print("Optimization Finished!")    
    

#####################################################
# FUNCTION TO READ ORIGINAL TRAIN DATAFRAME
#####################################################
def read_product_columns_train_dataframe():

    """ Reads train dataframe and converts types to reduce memory
    """
    df_train = pd.read_csv('train_ver2.csv', usecols = constants.PRODUCT_COLUMNS + ['fecha_dato', 'ncodpers'])

    df_train.ind_nomina_ult1.fillna(0, inplace=True)
    df_train.ind_nom_pens_ult1.fillna(0, inplace=True)

    for c in df_train.columns:
        if (c != 'renta') and (c != 'ncodpers') and ((df_train[c].dtype == np.int64) or (df_train[c].dtype == np.float64)):
            print(c)
            df_train[c] = df_train[c].astype(np.int16)

    df_train['fecha_dato'] = pd.to_datetime(df_train['fecha_dato'])
    return df_train


#####################################################
# FUNCTION TO GENERATE TEST PREDICTIONS
#####################################################
def compute_test_predictions(dataset, df_test, checkpoint_path, model, submission_path):
    """ Generate test predictions to be evaluated in kaggle
    dataset: dataset object which contains the test data ready to be feed in the model
    df_test: test original dataframe
    checkpoint_path: path of the model we want to use to generate the predictions
    model: tensorflow model (graph definition has to be created)
    submission_path: where to save the submission file
    """
    predictions_str_list = []
    ncodpers_test = []

    saver = tf.train.Saver()
    # Load some pickles of predictions with users with no interactions
    with open("preprocessed/previous_portfolio.pickle", 'rb') as handle:
        previous_portfolio = pickle.load(handle)
    with open("preprocessed/fixed_predictions_freqbaseline2.pickle", 'rb') as handle:
        fixed_predictions = pickle.load(handle)
        
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)

        for ncodpers in df_test.ncodpers.values:
            pred_str = ""
            # if we have past data, use it to obtain predictions
            if ncodpers in dataset['test']['ncodpers']:
                idx = dataset['test']['ncodpers'].index(ncodpers)
                batch_x_test = np.expand_dims(dataset['test']['X'][idx].toarray(), axis=0)
                predictions_test = sess.run(model.pred_prob, feed_dict={model.x: batch_x_test, model.dropout_keep_prob: 1})
                idx_predictions = np.arange(len(constants.PRODUCT_COLUMNS))
                sorted_pred, sorted_prods = zip(*sorted(zip(predictions_test[0], idx_predictions), reverse=True))
                num_added = 0
                for prob,prod in zip(sorted_pred, sorted_prods):
                    # If product was not part of portfolio in last month, add it as prediction
                    if previous_portfolio[ncodpers][prod] == 0:
                        pred_str += constants.PRODUCT_COLUMNS[prod] + ' '
                        num_added += 1
                    if num_added == 7:
                        break
            # if we don't have data, add the most likely products
            else:
                pred_str = fixed_predictions[ncodpers]
            ncodpers_test.append(ncodpers)
            predictions_str_list.append(pred_str)
            if len(ncodpers_test) % 10000 == 0:
                print(len(ncodpers_test))
                sys.stdout.flush()
                
    df_submit = pd.DataFrame()
    df_submit['ncodpers'] = ncodpers_test
    df_submit['added_products'] = predictions_str_list      

    df_submit.to_csv(submission_path, index=False)