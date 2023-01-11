# text_as_sequences

For Study text as sequence model input and predictions, words and speeches composed of sounds and rhythms you can do both word and syllables compared for most matching words in the sentence this is the only example they are asking me about how to build the ```text_as_sequence model``` process.

## Train ngram_model ##

To train networks model for working in learning and prediction from the inputs alphabest as sets of sequences, predefined and direct text to sequence relationship and support the initail value you can saved and load. In Example, remote devices none OS implementation.

```
def train_ngram_model(data, learning_rate=0.002, epochs=10, batch_size=32, layers=2, units=64, dropout_rate=0.5, 
        num_classes=2, vocab=vocab):
        
    embedding_layer = tf.keras.layers.Embedding(input_vocab_size, embedding_size)
    decoder_cell = tf.keras.layers.LSTMCell(hidden_size)
    sampler = tfa.seq2seq.TrainingSampler()
    output_layer = tf.keras.layers.Dense(output_vocab_size)
    decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer)

    input_ids = tf.random.uniform(
    [n_blocks, n_sizes], maxval=input_vocab_size, dtype=tf.int64)

    layer = tf.keras.layers.StringLookup(vocabulary=vocab)
    input_ids = layer(data)

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
```

## Create model ##

To initialize networks model for learning and prediction tasks, using Dense layer for simulation and LSTM for sequence to sequence process.

```
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
```

## Model prediction ##

Model prediction method, you can input the initail values or reset for new word prediction.

```
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
```

## Working logicals ##

Create data input as Tensorflow constant and training model, prediction with different inputs and see the result are related vary by the input different.

```
data = tf.constant([["a", "c", "d", "e", "d", "z", "b"], ["c", "d", "e", "d", "z", "b", "a"], ["d", "e", "d", "z", "b", "a", "c"], 
    ["e", "d", "z", "b", "a", "c", "d"]])
dataset = train_ngram_model( data, learning_rate, epochs, batch_size, layers, units, dropout_rate, num_classes )
model = model_initialize( n_blocks=7, n_sizes=4 )
model.fit( dataset, epochs=epochs, batch_size=1, callbacks=[custom_callback] )

##########################
data = tf.constant([["r", "s", "t", "u", "v", "w", "x"], ["r", "s", "t", "u", "v", "w", "x"], ["r", "s", "t", "u", "v", "w", "x"], 
    ["r", "s", "t", "u", "v", "w", "x"]])
result = target_prediction( data, model, n_blocks=7, n_sizes=4, input_vocab_size=128, vocab=vocab )
print( "result = " + str(result) )		# result = [[-0.09399483 -0.10728753  0.3197144   0.43001056]]

data = tf.constant([["a", "c", "d", "e", "d", "z", "b"], ["r", "s", "t", "u", "v", "w", "x"], ["e", "d", "z", "b", "a", "c", "d"], 
    ["c", "d", "e", "d", "z", "b", "a"]])
result = target_prediction( data, model, n_blocks=7, n_sizes=4, input_vocab_size=128, vocab=vocab )
print( "result = " + str(result) )		# result = [[-0.09399483 -0.10728753  0.3197144   0.43001056]]

data = tf.constant([["a", "c", "d", "e", "d", "z", "b"], ["c", "d", "e", "d", "z", "b", "a"], ["d", "e", "d", "z", "b", "a", "c"], 
    ["e", "d", "z", "b", "a", "c", "d"]])
result = target_prediction( data, model, n_blocks=7, n_sizes=4, input_vocab_size=128, vocab=vocab )
print( "result = " + str(result) )
```

## Result ##

Result in number as different from the training input it is small value but scales to more values.

```
result = tf.Tensor(0.25167924, shape=(), dtype=float32)
result = tf.Tensor(0.31068712, shape=(), dtype=float32)
result = tf.Tensor(0.31570023, shape=(), dtype=float32)
```

## Files and Directory ##

| file name | description |
| --- | --- |
| sample.py | sample codes |
| README.md | redme file |
