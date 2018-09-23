import tensorflow as tf
import torch
import numpy as np

import config, utils
from config import device, model as model_config
from sequence import EventSeq, Control, ControlSeq
from tensorflow_melody_rnn.model import CharRNN
import os
from IPython import embed

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('lstm_size', 512, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 3, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_string('control',
'../dataset/processed/drums/bach_850_format0.mid-f3ae883fc3c0e54dc5a7378740db5995.data',
                       'model/name/converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', './model/default', 'checkpoint path')
tf.flags.DEFINE_string('start_string', '', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 1000, 'max length to generate')



def main(_):
    if os.path.isfile(FLAGS.control) or os.path.isdir(FLAGS.control):
        if os.path.isdir(FLAGS.control):
            files = list(utils.find_files_by_extensions(FLAGS.control))
            assert len(files) > 0, 'no file in "{control}"'.format(control=FLAGS.control)
            control = np.random.choice(files)
        _, compressed_controls = torch.load(FLAGS.control)
        controls = ControlSeq.recover_compressed_array(compressed_controls)
        max_len = FLAGS.max_length
        if FLAGS.max_length == 0:
            max_len = controls.shape[0]

        control = np.expand_dims(controls, 1).repeat(1, 1)
        control = 'control sequence from "{control}"'.format(control=control)

    #assert max_len > 0, 'either max length or control sequence length should be given'

    #FLAGS.start_string = FLAGS.start_string.decode('utf-8')

    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path =\
            tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    model = CharRNN(EventSeq.dim(),ControlSeq.dim(), sampling=True,
                    lstm_size=FLAGS.lstm_size, num_layers=FLAGS.num_layers,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size)
    model.sess.run(tf.global_variables_initializer())
    model.load(FLAGS.checkpoint_path)

    outputs = model.sample(1000, prime =[60], vocab_size=EventSeq.dim())




    outputs = outputs.reshape([-1,1])
    print(outputs)
    name = 'output-{i:03d}.mid'.format(i=0)
    path = os.path.join("output/", name)
    n_notes = utils.event_indeces_to_midi_file(outputs[:,0], path)
    print('===> {path} ({n_notes} notes)'.format(path=path, n_notes=n_notes))


if __name__ == '__main__':
    tf.app.run()