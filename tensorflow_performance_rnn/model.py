import tensorflow as tf
import numpy as np
import time
import os
import tensorflow.contrib as contr


class Performance_RNN:
    def __init__(self, event_dim, control_dim, control = True, num_seqs=64, num_steps=50,
                 lstm_size=512, num_layers=2, learning_rate=0.001, greedy = 1.0,
                 grad_clip=5, sampling=False, train_keep_prob=0.5, use_embedding=False, embedding_size=128, temperature = 1.0):
        if sampling is True:
            num_seqs, num_steps = 1, 1
        else:
            num_seqs, num_steps = num_seqs, num_steps

        self.control = control #if use control

        self.event_dim = event_dim
        self.control_dim = control_dim

        self.num_seqs = num_seqs
        self.num_steps = num_steps -1
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size

        self.greedy =greedy
        self.temperature = temperature

        tf.reset_default_graph()
        self.build_inputs()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.event_inputs = tf.placeholder(tf.int32, shape=(
                None, self.num_steps), name='event_inputs')
            self.control_inputs = tf.placeholder(tf.float32, shape=(
                None, self.num_steps,self.control_dim), name='control_inputs')
            self.event_targets = tf.placeholder(tf.int32, shape=(
                None, self.num_steps), name='event_targets')

            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            with tf.device("/cpu:0"):
                embedding = tf.get_variable('embedding', [self.event_dim, self.event_dim])
                self.embedding_state = tf.nn.embedding_lookup(embedding, self.event_inputs)
            if self.control is False:
                default = tf.ones([self.num_seqs, self.num_steps, 1])
                self.control_inputs = tf.zeros([self.num_seqs, self.num_steps , self.control_dim])
            else:
                default = tf.zeros([self.num_seqs, self.num_steps, 1])

            self.inputs = tf.concat((self.embedding_state, default,self.control_inputs),axis=2)
            self.lstm_inputs = contr.layers.fully_connected(self.inputs, self.lstm_size, activation_fn= tf.nn.leaky_relu)

    def _sample_event(self, output, greedy=True, temperature=1.0):
        if greedy:
            return tf.argmax(output,axis=-1)
        else:
            output = output / temperature
            probs = tf.nn.softmax(output)
            return tf.distributions.Categorical(probs = probs).sample(self.event_dim)

    def build_lstm(self):
        # 创建单个cell并堆叠多层
        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
            )
            self.initial_state = cell.zero_state(self.num_seqs, tf.float32)

            # 通过dynamic_rnn对cell展开时间维度
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs, initial_state=self.initial_state)

            # 通过lstm_outputs得到概率
            seq_output = tf.concat(self.lstm_outputs, 1)
            x = tf.reshape(seq_output, [-1, self.lstm_size])

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.event_dim], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(self.event_dim))

            self.logits = tf.matmul(x, softmax_w) + softmax_b

            use_greedy = np.random.random() < self.greedy

            self.event_output = self._sample_event(self.logits, use_greedy, self.temperature)

    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.event_targets, self.event_dim)
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)

    def build_optimizer(self):
        # 使用clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def train(self, batch_generator, max_steps, save_path, save_every_n, log_every_n):
        with self.sess as sess:

            # Train network
            step = 0
            new_state = sess.run(self.initial_state)
            for event, control in batch_generator:

                step += 1
                start = time.time()
                target = event[:, 1:]
                feed = {self.event_inputs: event[:,0:-1],
                        self.control_inputs: control[:,0:-1,:],
                        self.event_targets: target,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([self.loss,
                                                     self.final_state,
                                                     self.optimizer],
                                                    feed_dict=feed)

                end = time.time()
                # control the print lines
                if step % log_every_n == 0:
                    print('step: {}/{}... '.format(step, max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if (step % save_every_n == 0):
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                if step >= max_steps:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)

    def sample(self, n_samples, control, vocab_size):

        def pick_top_n(preds, vocab_size, top_n=50):
            p = np.squeeze(preds)
            # 将除了top_n个预测值的位置都置为0
            p[np.argsort(p)[:-top_n]] = 0
            # 归一化概率
            p = p / np.sum(p)
            # 随机选取一个字符
            c = np.random.choice(vocab_size, 1, p=p)[0]
            if np.random.uniform()<0.85:
                c = np.argmax(preds)
            return c

        prime = np.random.choice(self.event_dim,size=1)[0]
        samples = [prime]
        sess = self.sess
        new_state = sess.run(self.initial_state)
        preds = np.ones((vocab_size, ))  # for prime=[]


        feed = {self.event_inputs: [[prime]],
                self.control_inputs: [[control[0]]],
                self.keep_prob: 1.,
                self.initial_state: new_state}
        preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                    feed_dict=feed)

        c = pick_top_n(preds, vocab_size)
        # 添加字符到samples中
        samples.append(c)

        # 不断生成字符，直到达到指定数目
        for i in range(1, n_samples):

            feed = {self.event_inputs: [[c]],
                    self.control_inputs: [[control[i]]],
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, vocab_size)
            samples.append(c)

        return np.array(samples)

    def load(self, checkpoint):
        self.saver.restore(self.sess, checkpoint)
        print('Restored from: {}'.format(checkpoint))
