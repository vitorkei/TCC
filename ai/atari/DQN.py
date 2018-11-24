import tensorflow as tf

class DQNNet:
  def __init__(self, state_size, action_size, optimizer_params, conv_params, units, name):
    self.state_size       = state_size
    self.action_size      = action_size
    self.optimizer_params = optimizer_params
    self.conv_filters     = conv_params[0]
    self.kernel_sizes     = conv_params[1]
    self.stride_sizes     = conv_params[2]
    self.units            = units
    self.name             = name

    with tf.variable_scope(self.name):
      self.input    = tf.placeholder(tf.float32, [None, *state_size])
      self.actions  = tf.placeholder(tf.float32, [None, self.action_size])
      self.Q_target = tf.placeholder(tf.float32, [None])

      self.conv2d = tf.layers.conv2d(inputs             = self.input,
                                     filters            = self.conv_filters[0],
                                     kernel_size        = self.kernel_sizes[0],
                                     strides            = self.stride_sizes[0],
                                     activation         = tf.nn.relu,
                                     use_bias           = False,
                                     kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
                                    )

      for i in range(1, len(self.conv_filters)):
        self.conv2d = tf.layers.conv2d(inputs             = self.conv2d,
                                       filters            = self.conv_filters[i],
                                       kernel_size        = self.kernel_sizes[i],
                                       strides            = self.stride_sizes[i],
                                       activation         = tf.nn.relu,
                                       use_bias           = False,
                                       kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
                                      )

      self.fc = tf.layers.dense(inputs             = tf.layers.flatten(self.conv2d),
                                units              = self.units,
                                activation         = tf.nn.relu,
                                kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
                                )

      self.output = tf.layers.dense(inputs             = self.fc,
                                    units              = self.action_size,
                                    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
                                   )
      
      self.Q         = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)
      self.loss      = tf.losses.huber_loss(labels      = self.Q_target,
                                            predictions = self.Q)
      self.optimizer = tf.train.RMSPropOptimizer(learning_rate = self.optimizer_params[0],
                                                 momentum      = self.optimizer_params[1],
                                                 epsilon       = self.optimizer_params[2]).minimize(self.loss)
