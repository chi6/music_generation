import tensorflow as tf
from data import Dataset
from tensorflow_melody_rnn.model import CharRNN
import os
import codecs
from sequence import NoteSeq, EventSeq, ControlSeq

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name', 'default', 'name of the model')
tf.flags.DEFINE_integer('num_seqs', 64, 'number of seqs in one batch')
tf.flags.DEFINE_integer('num_steps', 200, 'length of one seq')
tf.flags.DEFINE_integer('lstm_size', 512, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 3, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.flags.DEFINE_string('data_path', '../dataset/processed/drums', 'data_path')
tf.flags.DEFINE_integer('max_steps', 100000, 'max steps to train')
tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')
tf.flags.DEFINE_integer('log_every_n', 10, 'log to the screen every n steps')
tf.flags.DEFINE_integer('max_vocab', 3500, 'max char number')

def load_dataset():
    dataset = Dataset(FLAGS.data_path, verbose=True)
    dataset_size = len(dataset.samples)
    assert dataset_size > 0
    return dataset

def main(_):
    model_path = os.path.join('./model/', FLAGS.name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    if os.path.isdir(model_path):
        checkpoint_path =\
            tf.train.latest_checkpoint(model_path)

    dataset = load_dataset()
    batch_gen = dataset.batches(FLAGS.num_seqs, FLAGS.num_steps, 10)
    print(EventSeq.dim())
    model = CharRNN(EventSeq.dim(),ControlSeq.dim(),
                    num_seqs=FLAGS.num_seqs,
                    num_steps=FLAGS.num_steps,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size
                    )
    model.sess.run(tf.global_variables_initializer())
    model.load(checkpoint_path)


    model.train(batch_gen,
                FLAGS.max_steps,
                model_path,
                FLAGS.save_every_n,
                FLAGS.log_every_n,
                )


if __name__ == '__main__':
    tf.app.run()