import os
import tensorflow as tf
from model import Model
from loader import load_data
from utils import BatchManager, test_ner        

#import sys
#from imp import reload
#reload(sys)
#sys.setdefaultencoding('utf-8')

FLAGS = tf.app.flags.FLAGS
# path for log, model and result
tf.app.flags.DEFINE_string("log_path", './jian/log', "path for log files")
tf.app.flags.DEFINE_string("model_path", './jian/models', "path to save model")
tf.app.flags.DEFINE_string("result_path", './jian/results', "path to save result")
tf.app.flags.DEFINE_string("train_file", './data/train.jian', "path for train data")
#tf.app.flags.DEFINE_string("train_file", './data/SIGHAN.NER.train', "path for train data")
#tf.app.flags.DEFINE_string("dev_file", './data/SIGHAN.NER.dev', "path for valid data")
#tf.app.flags.DEFINE_string("test_file", './data/SIGHAN.NER.test', "path for test data")
tf.app.flags.DEFINE_string("dev_file", './data/test.jian', "path for valid data")
tf.app.flags.DEFINE_string("test_file", './data/pred.jian', "path for test data")
# config for model
tf.app.flags.DEFINE_boolean("lower", True, "True for lowercase all characters")
tf.app.flags.DEFINE_string("pre_emb", "./embedding/wiki_word2vec.pkl",
                           "path for pre-trained embedding, False for randomly initialize")
tf.app.flags.DEFINE_integer("min_freq", 1, "")
tf.app.flags.DEFINE_integer("word_max_len", 100, "maximum words in a sentence")
tf.app.flags.DEFINE_integer("word_dim", 100, "dimension of char embedding")
tf.app.flags.DEFINE_integer("word_hidden_dim", 150, "dimension of word LSTM hidden units")
tf.app.flags.DEFINE_string("feature_dim", 4, "dimension of extra features, 0 for not used")
# config for training process

tf.app.flags.DEFINE_float("dropout", 0.5, "dropout rate")
tf.app.flags.DEFINE_float("clip", 5, "gradient to clip")
tf.app.flags.DEFINE_float("lr", 0.001, "initial learning rate")
tf.app.flags.DEFINE_integer("max_epoch", 100, "maximum training epochs")
tf.app.flags.DEFINE_integer("batch_size", 200, "num of sentences per batch")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100, "steps per checkpoint")
tf.app.flags.DEFINE_integer("valid_batch_size", 100, "num of sentences per batch")


def create_model(session, word_to_id, id_to_tag):
    # create model, reuse parameters if exists
    model = Model("tagger", word_to_id, id_to_tag, FLAGS)

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        model.logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        model.logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def main(_):
    if not os.path.isdir(FLAGS.log_path):
        os.makedirs(FLAGS.log_path)
    if not os.path.isdir(FLAGS.model_path):
        os.makedirs(FLAGS.model_path)
    if not os.path.isdir(FLAGS.result_path):
        os.makedirs(FLAGS.result_path)
    tag_to_id = {"O": 0, "B-LOC": 1, "I-LOC": 2,
                 "B-PER": 3, "I-PER": 4, "B-ORG": 5, "I-ORG": 6}
    # load data
    word_to_id, id_to_tag, train_data, dev_data, test_data = load_data(FLAGS, tag_to_id)
    
    test_manager = BatchManager(test_data, len(id_to_tag), FLAGS.word_max_len, FLAGS.valid_batch_size)
    with tf.Session() as sess:
        model = create_model(sess, word_to_id, id_to_tag)

#        return
        # test model
#        model.logger.info("testing ner")
#        ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
#        model.logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
#        model.saver.restore(sess, ckpt.model_checkpoint_path)
        ner_results = model.predict(sess, test_manager)         # batch nums
        
        print ('test data lengths: ' + str(len(test_manager.data)))
        print ('test batch size: ' + str(test_manager.batch_size))
        print ('test result lengths: ' + str(len(ner_results)))
        print (ner_results[0])
        eval_lines = test_ner(ner_results, FLAGS.result_path)
        for line in eval_lines:
            model.logger.info(line)


if __name__ == "__main__":
    tf.app.run(main)
