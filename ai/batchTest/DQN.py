# Classe Deep Q-Network que implementa deep Q-learning

import tensorflow as tf

class DQN:
  def __init__(self, state_size, action_size, name, config):
    self.state_size = state_size
    self.action_size = action_size
    self.name = name
    self.config = config

    with tf.variable_scope(self.name):
      # O espaço None é um placeholder para o tamanho do mini-batch
      self.input = tf.placeholder(tf.float32, [None, *state_size])
      self.action = tf.placeholder(tf.float32, [None, self.action_size])
      self.target_Q = tf.placeholder(tf.float32, [None])

      # Camadas de convolução + ReLU
      self.conv2d = tf.layers.conv2d(inputs=self.input,
                                     filters=self.config.conv_filters[0],
                                     kernel_size=self.config.kernel_sizes[0],
                                     strides=self.config.stride_sizes[0],
                                     activation=tf.nn.relu,
                                     use_bias=False,
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                    )
      for i in range(1, len(self.config.conv_filters)):
        self.conv2d = tf.layers.conv2d(inputs=self.conv2d,
                                       filters=self.config.conv_filters[i],
                                       kernel_size=self.config.kernel_sizes[i],
                                       strides=self.config.stride_sizes[i],
                                       activation=tf.nn.relu,
                                       use_bias=False,
                                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                      )

      # Camadas fully-connected
      self.flatten = tf.layers.flatten(self.conv2d)
      self.fc = tf.layers.dense(inputs=self.flatten,
                                units=self.config.units,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                               )
      self.output = tf.layers.dense(inputs=self.fc,
                                    units=self.action_size,
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                   )

      # Cálculo do erro e minimização
      self.Q = tf.reduce_sum(tf.multiply(self.output, self.action), axis=1)
      self.loss = tf.losses.huber_loss(labels=self.target_Q,
                                       predictions=self.Q)
      self.optimizer = tf.train.RMSPropOptimizer(self.config.learning_rate,
                                                 momentum=self.config.momentum,
                                                 epsilon=self.config.epsilon
                                                ).minimize(self.loss)
