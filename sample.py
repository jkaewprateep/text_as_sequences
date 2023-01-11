# https://stackoverflow.com/questions/72020939/valueerror-logits-and-labels-must-have-the-same-shape-received-none-250
# ValueError: `logits` and `labels` must have the same shape, received ((None, 250, 1) vs (None,)). What is wrong?
#
# Input: (['10@¨260 :?Kings .]~H.wy ','3109 n drake' '(1`72¿0" |3¥4®th St SE'],['something else','something else 2'],[1,0])
# Error: ValueError: `logits` and `labels` must have the same shape, received ((None, 250, 1) vs (None,)).



import tensorflow as tf
import tensorflow_addons as tfa

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
epochs = 1000
# learning_rate = 0.002
learning_rate = 0.00002
batch_size = 4
layers = 2
units = 64
dropout_rate = 0.5
# num_classes = 2
num_classes = 2
input_vocab_size = 128
output_vocab_size = 64
embedding_size = 48
hidden_size = 32
max_time = 7
batch_size = 1
n_blocks = 7
n_sizes = 4

vocab = ["a", "b", "c", "d", "e", "f", "g"]
model = tf.keras.models.Sequential([ ])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Callback
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class custom_callback(tf.keras.callbacks.Callback):

	def on_epoch_end(self, epoch, logs={}):
		if(logs['accuracy'] >= 0.97) : 
			self.model.stop_training = True

custom_callback = custom_callback( )

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Definition
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def train_ngram_model(data, learning_rate=0.002, epochs=10, batch_size=32, layers=2, units=64, dropout_rate=0.5, num_classes=2, vocab=vocab):

	embedding_layer = tf.keras.layers.Embedding(input_vocab_size, embedding_size)
	###
	decoder_cell = tf.keras.layers.LSTMCell(hidden_size)
	sampler = tfa.seq2seq.TrainingSampler()
	output_layer = tf.keras.layers.Dense(output_vocab_size)
	decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer)

	########################## 
	input_ids = tf.random.uniform(
	[n_blocks, n_sizes], maxval=input_vocab_size, dtype=tf.int64)

	layer = tf.keras.layers.StringLookup(vocabulary=vocab)
	input_ids = layer(data)
	##########################    

	input_lengths = tf.fill([batch_size], max_time)
	input_tensors = embedding_layer(input_ids)
	initial_state = decoder_cell.get_initial_state(input_tensors)

	output, state, lengths = decoder( input_tensors, sequence_length=input_lengths, initial_state=initial_state )
	logits = output.rnn_output

	input_ids = tf.cast( input_ids, dtype=tf.float32 )
	input_ids = tf.constant( input_ids, shape=(1, 1, n_blocks, n_sizes), dtype=tf.float32 )

	label = tf.constant([[0.], [1.], [2.], [3.]], shape=( 1, 1, 1, n_sizes ))

	dataset = tf.data.Dataset.from_tensor_slices(( input_ids, label ))

	return dataset

def model_initialize( n_blocks=7, n_sizes=4 ):

    model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(n_blocks, n_sizes)),

	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, return_state=False)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
	
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(192, activation='relu'),
    tf.keras.layers.Dense(4),
    ])
            
    model.summary()
    
    model.compile( loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy'] )
    
    return model

def target_prediction( data, model, n_blocks=7, n_sizes=4, input_vocab_size=128, vocab=vocab ):

	########################## 
	input_ids = tf.random.uniform(
	[n_blocks, n_sizes], maxval=input_vocab_size, dtype=tf.int64)

	layer = tf.keras.layers.StringLookup(vocabulary=vocab)
	input_ids = layer(data)
	##########################    

	prediction_input =  tf.cast( input_ids, dtype=tf.float32 )
	prediction_input = tf.constant( prediction_input, shape=( 1, n_blocks, n_sizes ), dtype=tf.float32 )
	predictions = model.predict( prediction_input )
	result = tf.keras.layers.Softmax()(predictions[0])
	result_arg = tf.math.argmax(result).numpy()
	result = result[int(result_arg)]

	return result

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Working logicals
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
data = tf.constant([["a", "c", "d", "e", "d", "z", "b"], ["c", "d", "e", "d", "z", "b", "a"], ["d", "e", "d", "z", "b", "a", "c"], ["e", "d", "z", "b", "a", "c", "d"]])
dataset = train_ngram_model( data, learning_rate, epochs, batch_size, layers, units, dropout_rate, num_classes )
model = model_initialize( n_blocks=7, n_sizes=4 )
model.fit( dataset, epochs=epochs, batch_size=1, callbacks=[custom_callback] )

##########################
data = tf.constant([["r", "s", "t", "u", "v", "w", "x"], ["r", "s", "t", "u", "v", "w", "x"], ["r", "s", "t", "u", "v", "w", "x"], ["r", "s", "t", "u", "v", "w", "x"]])
result = target_prediction( data, model, n_blocks=7, n_sizes=4, input_vocab_size=128, vocab=vocab )
print( "result = " + str(result) )		# result = [[-0.09399483 -0.10728753  0.3197144   0.43001056]]

data = tf.constant([["a", "c", "d", "e", "d", "z", "b"], ["r", "s", "t", "u", "v", "w", "x"], ["e", "d", "z", "b", "a", "c", "d"], ["c", "d", "e", "d", "z", "b", "a"]])
result = target_prediction( data, model, n_blocks=7, n_sizes=4, input_vocab_size=128, vocab=vocab )
print( "result = " + str(result) )		# result = [[-0.09399483 -0.10728753  0.3197144   0.43001056]]

data = tf.constant([["a", "c", "d", "e", "d", "z", "b"], ["c", "d", "e", "d", "z", "b", "a"], ["d", "e", "d", "z", "b", "a", "c"], ["e", "d", "z", "b", "a", "c", "d"]])
result = target_prediction( data, model, n_blocks=7, n_sizes=4, input_vocab_size=128, vocab=vocab )
print( "result = " + str(result) )

input('...')
