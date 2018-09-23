import tensorflow as tf
import numpy as np
import time
import os



class CharRNN:
    def __init__(self, event_dim, control_dim, num_seqs=64, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False, train_keep_prob=0.5, use_embedding=False, embedding_size=128):
        if sampling is True:
            num_seqs, num_steps = 1, 500
        else:
            num_seqs, num_steps = num_seqs, num_steps

        self.event_dim = event_dim
        self.control_dim = control_dim

        self.num_seqs = num_seqs
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size

        tf.reset_default_graph()
        self.build_inputs()
        self.build_lstm()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()


    def _build_lstm_cell(self):
        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size,state_is_tuple=False)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop
        cell = tf.contrib.rnn.MultiRNNCell([get_a_cell(self.lstm_size,self.keep_prob) for _ in range(self.num_layers)])
        return  cell

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.classical_inputs = tf.placeholder(tf.int32, shape=(
                self.num_seqs, self.num_steps), name='classical_inputs')
            self.drum_inputs = tf.placeholder(tf.int32, shape=(
                self.num_seqs, self.num_steps), name='drum_inputs')
            self.classical_targets = tf.placeholder(tf.int32, shape=(
                self.num_seqs, self.num_steps), name='classical_targets')
            self.drum_targets = tf.placeholder(tf.int32, shape=(
                self.num_seqs, self.num_steps), name='drum_targets')
            self.decoder_targets_length = tf.placeholder(tf.int32, [None], name= 'decoder_targets_lenght')

            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')

            self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
            self.mask = tf.sequence_mask(self.decoder_targets_length, self.max_target_sequence_length, dtype=tf.float32,
                                         name='masks')
            self.encoder_inputs_length =  tf.placeholder(tf.int32, [None], name='encoder_inputs_length')

            with tf.device("/cpu:0"):
                self.embedding = tf.get_variable('embedding', [self.event_dim, self.event_dim])
                self.classical_lstm_inputs = tf.nn.embedding_lookup(self.embedding, self.classical_inputs)

                self.drum_lstm_inputs = tf.nn.embedding_lookup(self.embedding, self.drum_inputs)

    def build_lstm(self):

        with tf.name_scope('classical_encoder'):
            cell = self._build_lstm_cell()
            self.initial_state = cell.zero_state(self.num_seqs, tf.float32)

            self.classical_encoder_outputs, self.classical_encoder_state = tf.nn.dynamic_rnn(cell, self.classical_lstm_inputs,
                                                                   dtype=tf.float32)

            W_mu = tf.Variable(tf.truncated_normal([self.lstm_size , 128], stddev=0.1))
            b_mu = tf.Variable(tf.zeros(128))
            W_log_sigma = tf.Variable(tf.truncated_normal([self.lstm_size, 128], stddev=0.1))
            b_log_sigma = tf.Variable(tf.zeros(128))

            # using the encoded c state of the latest lstm

            z_mu = tf.matmul(self.classical_encoder_state[-1][:,0:self.lstm_size], W_mu) + b_mu
            z_log_sigma = (tf.matmul(self.classical_encoder_state[-1][:,0:self.lstm_size], W_log_sigma) + b_log_sigma)
            epsilon = tf.random_normal(
                shape=(self.num_seqs, 128))
            # encode in a (1, 128) shape latent code for each batch
            z = z_mu + tf.exp(z_log_sigma)*epsilon
            # output shape of lstm hidden_size
            W_dec1 =tf.Variable(tf.truncated_normal([128, self.lstm_size], stddev=0.1))
            b_dec1 = tf.Variable(tf.zeros(self.lstm_size))

            h_dec1 = tf.nn.relu(tf.add(tf.matmul(z, W_dec1), b_dec1))


        with tf.variable_scope('decoder'):
            # define decoder
            encoder_inputs_length = self.encoder_inputs_length
            decoder_cell = self._build_lstm_cell()

            batch_size = self.batch_size

            #decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            output_layer = tf.layers.Dense(self.event_dim,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            #train
            ending = tf.strided_slice(self.classical_targets, [0, 0], [self.batch_size, -1], [1, 1])
            decoder_input = tf.concat([tf.fill([self.num_seqs, 1], 1), ending], 1)
            decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, decoder_input)

            initial_state = (tf.concat((h_dec1,self.classical_encoder_state[0][:,self.lstm_size::]),1),
                             tf.concat((h_dec1, self.classical_encoder_state[1][:, self.lstm_size::]), 1),
                                tf.concat((h_dec1, self.classical_encoder_state[2][:, self.lstm_size::]), 1))

            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
                                                                sequence_length=self.decoder_targets_length,
                                                                time_major=False, name='training_helper')
            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=training_helper,
                                                               initial_state=initial_state,
                                                               output_layer=output_layer)
            # 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，里面包含两项(rnn_outputs, sample_id)
            # rnn_output: [batch_size, decoder_targets_length, vocab_size]，保存decode每个时刻每个单词的概率，可以用来计算loss
            # sample_id: [batch_size], tf.int32，保存最终的编码结果。可以表示最后的答案
            decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                      impute_finished=True,
                                                                      maximum_iterations=self.max_target_sequence_length)

            self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
            self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis=2, name='decoder_pred_train')
            print(self.classical_encoder_outputs)
            # 使用sequence_loss计算loss，这里需要传入之前定义的mask标志
            kl_div = -0.5 * tf.reduce_sum(
                1.0 + 2.0 * z_log_sigma - tf.square(z_mu) - tf.exp(2.0 * z_log_sigma),
                1)

            self.loss1 = tf.reduce_mean(kl_div)
            self.loss2 = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                         targets=self.classical_targets, weights=self.mask))

            self.loss = self.loss1 + 40*self.loss2
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))


    def train(self, classical_batch_generator, drum_batch_generator,
              max_steps, save_path, save_every_n, log_every_n):
        with self.sess as sess:
            # Train network
            step = 0
            print(self.num_steps)
            encoder_target_length = [self.num_steps for i in range(self.num_seqs)]
            decoder_targets_length = [self.num_steps for i in range(self.num_seqs)]
            for x, x1 in zip(classical_batch_generator,drum_batch_generator):
                step += 1
                start = time.time()
                inputs = x[:,int(self.num_steps)::]

                target = x[:,0:int(self.num_steps)]
                feed_dict = {self.classical_inputs: inputs,
                             self.encoder_inputs_length: encoder_target_length,
                             self.classical_targets: target,
                             self.decoder_targets_length: decoder_targets_length,
                             self.keep_prob: 0.5,
                             self.batch_size: self.num_seqs}
                _, loss, loss1,loss2 = sess.run([self.train_op, self.loss,self.loss1,self.loss2], feed_dict=feed_dict)


                end = time.time()
                # control the print lines
                if step % log_every_n == 0:
                    print('step: {}/{}... '.format(step, max_steps),
                          'loss: {:.4f}... '.format(loss),
                          'loss_kl: {:.4f}... '.format(loss1),
                          'loss_seq: {:.4f}... '.format(loss2),
                          '{:.4f} sec/batch'.format((end - start)))
                if (step % save_every_n == 0):
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                if step >= max_steps:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)

    def sample(self, n_samples,prime_classical, prime_drum=None, vocab_size=240):

        samples0 = [c for c in prime_classical]
        sess = self.sess

        encoder_target_length = [len(samples0)]
        decoder_targets_length = [len(samples0)]


        feed_dict = {self.classical_inputs: [prime_classical],
                     self.encoder_inputs_length: encoder_target_length,
                     self.decoder_targets_length: decoder_targets_length,
                     self.classical_targets:[prime_classical],
                     self.keep_prob: 0.6,
                     self.batch_size: 1}
        pred_ids = sess.run([self.decoder_predict_train], feed_dict=feed_dict)

        samples0 = [i for i in pred_ids[0][0]]

        while len(samples0)<n_samples:
            encoder_target_length = [len(prime_classical)]
            decoder_targets_length = [len(prime_classical)]

            feed_dict = {self.classical_inputs: pred_ids[0],
                         self.encoder_inputs_length: encoder_target_length,
                         self.decoder_targets_length: decoder_targets_length,
                         self.classical_targets: pred_ids[0],
                         self.keep_prob: 0.8,
                         self.batch_size: 1}
            pred_ids = sess.run([self.decoder_predict_train], feed_dict=feed_dict)
            samples0 += [i for i in pred_ids[0][0]]
        return samples0

    def load(self, checkpoint):
        self.saver.restore(self.sess, checkpoint)
        print('Restored from: {}'.format(checkpoint))
