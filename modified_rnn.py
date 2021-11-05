# import packages
import numpy as np
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# load training and test datafiles
def load_data():
    fn_train = "matlab_train1.mat"
    fn_test =  "matlab_test.mat"

    d1_idx=0           # data range of time axis from training file
    d1e_idx=100        # data range of time axis from training file
    d_idx=0            # data range of time axis from test file
    de_idx=100         # data range of time axis from test file
    num=100

    data_train= sio.loadmat(fn_train)
    y_train = np.zeros((num,1),np.int32)
    X_train=np.zeros((num,(d1e_idx-d1_idx),6))

    # loading training file and building a 3d feature matrix
    X_train[0,:,0]=data_train['App_r161'][0,d1_idx:d1e_idx]
    X_train[0,:,1]=data_train['EngT_t1'][0,d1_idx:d1e_idx]
    X_train[0,:,2]=data_train['Eng_N1'][0,d1_idx:d1e_idx]
    X_train[0,:,3]=data_train['Thr_r1'][0,d1_idx:d1e_idx]
    X_train[0,:,4]=data_train['Chrg_Ld1'][0,d1_idx:d1e_idx]
    X_train[0,:,5]=data_train['Chrg_LdDes1'][0,d1_idx:d1e_idx]

    # for generating training pattern as seen in the image after code execution
    for i in range(round(num/2)):
        X_train[2*i,:,:]=X_train[0,:,:]
        y_train[2*i, 0] = 9

    data_test = sio.loadmat(fn_test)
    y_test = np.zeros((10,1),np.int32)
    X_test = np.zeros((10,de_idx-d_idx,6))

    # loading test file and building a 3d feature matrix
    X_test[0,:,0] = data_test['App_r161'][0,d_idx:de_idx]
    X_test[0,:,1] = data_test['EngT_t1'][0,d_idx:de_idx]
    X_test[0,:,2] = data_test['Eng_N1'][0,d_idx:de_idx]
    X_test[0,:,3] = data_test['Thr_r1'][0,d_idx:de_idx]
    X_test[0,:,4] = data_test['Chrg_Ld1'][0,d_idx:de_idx]
    X_test[0,:,5] = data_test['Chrg_LdDes1'][0,d_idx:de_idx]

    for i in range(5):
        X_test[2*i,:,:]=X_train[0,:,:]
        y_test[2*i, 0] = 9

    return X_train, y_train,X_test,y_test


# Set the directory path
#file_name = os.path.abspath(os.path.join(os.path.curdir, 'data','trial','train data'));
#summaries_dir = '/home/warewolf/PycharmProjects/hiwi/data/log'

x_train, y_train,x_test,y_test = load_data()


training_data_count = len(x_train)  # 4 training series
test_data_count = len(x_test)  # 10 testing series
n_steps = len(x_train[0]) # 100 timesteps per series
n_input = len(x_train[0][0]) # 6 input parameters per timestep
n_hidden = 32 # Hidden layer num of features
n_classes = (np.max((np.max(y_train),np.max(y_test)))) + 1 # Total classes
learning_rate = 0.0025
lambda_loss_amount = 0.0015
training_iters = training_data_count * 300  # Loop 300 times on the dataset
batch_size = 1500
display_iter = 300  # To show test set accuracy during training


# rnn functionality
def LSTM_RNN(_X, _weights, _biases):

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input])
    # new shape: (n_steps*batch_size, n_input)

    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data.

    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step - 1) * batch_size + i) % len(_train)
        batch_s[i] = _train[index]

    return batch_s

def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))

    # n_values = int(np.max(y_)) + 1
    n_values=n_classes
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

# Graph input/output
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Graph weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = LSTM_RNN(x, weights, biases)

print("training",x_train.shape,y_train.shape)
print("testing",x_test.shape,y_test.shape)

# Loss, optimizer and evaluation
l2 = lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
# L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2 # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))  # checking for number of correct predictions
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# To keep track of training's performance
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []

# Launch the graph
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)
# Perform Training steps with "batch_size" amount of example data at each loop
step = 1
while step * batch_size <= training_iters:
    batch_xs = extract_batch_size(x_train, step, batch_size)
    batch_ys = one_hot(extract_batch_size(y_train, step, batch_size))
    # Fit training using batch data
    _, loss, acc = sess.run([optimizer, cost, accuracy],feed_dict={ x: batch_xs, y: batch_ys })
    train_losses.append(loss)
    train_accuracies.append(acc)

    # Evaluate network only at some steps for faster training:
    if (step * batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
        # To not spam console, show training accuracy/loss in this "if"
        print("Training iter #" + str(step * batch_size) + \
              ":   Batch Loss = " + "{:.6f}".format(loss) + \
              ", Accuracy = {}".format(acc))

        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        loss, acc = sess.run([cost, accuracy],feed_dict={x: x_test, y: one_hot(y_test)} )
        test_losses.append(loss)
        test_accuracies.append(acc)

        print("PERFORMANCE ON TEST SET: " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc))

    step += 1

print("Optimization Finished!")

# Accuracy for test data
one_hot_predictions= sess.run(pred,feed_dict={x: x_test,y: one_hot(y_test)})

# print("FINAL RESULT: " + \
#       "Batch Loss = {}".format(final_loss) + \
#       ", Accuracy = {}".format(accuracy))

plt.figure()
plt.subplot(211)
plt.xlabel('actual output')
plt.plot(y_test)

plt.subplot(212)
plt.plot(np.argmax(one_hot_predictions,1))
plt.xlabel('predicted output')

plt.show()
