# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import numpy as np
import tensorflow as tf
import time, os

class Logger():
    def __init__(self, sess, args, summary_names, key_output_index):
        self.sess = sess
        self.dir = args.dir
        self.log = args.log
        self.model_dir = os.path.join(self.dir, "model")
        self.log_dir = os.path.join(self.dir, "log")
        self.display_interval = args.display_interval

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.global_step_inc_op = self.global_step.assign(self.global_step + 1)    
        self.epoch = tf.Variable(1, name="epoch", trainable=False)
        self.epoch_inc_op = self.epoch.assign(self.epoch + 1)

        self.summary_names = summary_names
        self.summary_num = len(self.summary_names)
        self.key_output_index = key_output_index

        self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "train"))
        self.valid_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "valid"))
        self.test_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "test"))
        self.summary_placeholders = [tf.placeholder(tf.float32) for i in range(self.summary_num)]
        self.summary_op = [tf.summary.scalar(
            self.summary_names[i], self.summary_placeholders[i]) for i in range(self.summary_num)]

        params = []
        for var in tf.global_variables():
            params.append(var)

        self.saver = tf.train.Saver(
            params,
            write_version=tf.train.SaverDef.V2,
            max_to_keep=None, 
            pad_step_number=True, 
            keep_checkpoint_every_n_hours=1.0
        )        
        
        self.sess.run(tf.global_variables_initializer())
        if tf.train.get_checkpoint_state(self.model_dir):    
            print("Restoring model...")
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))

        with open(self.log, "w") as file: pass

        self.best = 0 
        self.best_valid = 0       
        self.decay = False
        self._clear()

    def _clear(self):
        self.s_train = np.zeros(self.summary_num)
        self.s_valid = np.zeros(self.summary_num)
        self.s_test = np.zeros(self.summary_num)
        self.train_steps = self.valid_steps = self.test_steps = 0
        self.start_time = time.time()

    def get_summary_sum(self, s, length):
        if length == 0: 
            return s
        else:
            return s / length
        
    def next_epoch(self):
        with self.sess.as_default():
            summary_sum = self.get_summary_sum(self.s_train, self.train_steps)            
            summaries = self.sess.run(self.summary_op, feed_dict=dict(zip(self.summary_placeholders, summary_sum)))
            for s in summaries:
                self.train_writer.add_summary(summary=s, global_step=self.epoch.eval())                        
            print("epoch", self.epoch.eval())
            print("  train", end="")
            for k in range(self.summary_num):
                print(" {}: {:.5f}".format(self.summary_names[k], summary_sum[k]), end="")
            print()
            self.epoch_inc_op.eval()
            self.saver.save(self.sess, os.path.join(self.model_dir, "ckpt"), global_step=self.epoch.eval()-1)   
               
    def next_step(self, out):
        self.train_steps += 1
        for i in range(min(self.summary_num, len(out))):
            self.s_train[i] += out[i]

        self.global_step_inc_op.eval()
        self.global_step_val = self.global_step.eval()         
        if self.global_step_val % self.display_interval == 0:
            print("epoch %d, global step %d (%.3fs/step):" % (
                self.epoch.eval(), self.global_step_val, 
                (time.time() - self.start_time) * 1. / self.train_steps
            ))
            summary_sum = self.get_summary_sum(self.s_train, self.train_steps)
            print("  best {:.5f}".format(self.best))
            print("  train", end="")
            for k in range(self.summary_num):
                print(" {} {:.5f}".format(self.summary_names[k], summary_sum[k]), end="")
            print()

    def add_valid(self, out):
        self.valid_steps += 1
        for i in range(min(self.summary_num, len(out))):
            self.s_valid[i] += out[i]

    def add_test(self, out):
        self.test_steps += 1
        for i in range(min(self.summary_num, len(out))):
            self.s_test[i] += out[i]       


    def save_valid(self, log=False):
        summary_sum = self.get_summary_sum(self.s_valid, self.valid_steps)
        if log:
            summaries = self.sess.run(self.summary_op, feed_dict=dict(zip(self.summary_placeholders, summary_sum)))
            for s in summaries:
                self.valid_writer.add_summary(summary=s, global_step=self.epoch.eval()-1)
        print("  valid", end="")
        for k in range(self.summary_num):
            print(" {}: {:.5f}".format(self.summary_names[k], summary_sum[k]), end="")
        print()
        self.valid_steps = 0
        self.s_valid = np.zeros(self.summary_num)
        if summary_sum[self.key_output_index] > self.best_valid:
            self.best_valid = summary_sum[self.key_output_index]
            self.decay = False
        else:
            self.decay = True

    def save_test(self, log=False):
        summary_sum = self.get_summary_sum(self.s_test, self.test_steps)
        if log:
            summaries = self.sess.run(self.summary_op, feed_dict=dict(zip(self.summary_placeholders, summary_sum)))
            for s in summaries:
                self.test_writer.add_summary(summary=s, global_step=self.epoch.eval()-1)
        print("  test", end="")
        for k in range(self.summary_num):
            print(" {}: {:.5f}".format(self.summary_names[k], summary_sum[k]), end="")
        print()
        if summary_sum[self.key_output_index] > self.best:
            self.best = summary_sum[self.key_output_index]
        print("  best: {:.5f}".format(self.best))
        self.test_steps = 0
        self.s_test = np.zeros(self.summary_num)
        if log:
            self._clear() 

    def get_epoch(self):
        return self.epoch.eval()

    def write(self, *all_text):
        with open(self.log, "a+") as file:
            for text in all_text:
                print(text, end=" ")
                file.write("{} ".format(text))
            print()
            file.write("\n")