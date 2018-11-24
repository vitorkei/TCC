import tensorflow as tf

class DQN:
  def __init__(self, state_size, action_size, optimizer_params, conv_params, name):
    self.state_size       = state_size
    self.action_size      = action_size
    self.optimizer_params = optimizer_params
    self.conv_filters     = conv_params[0]
    self.kernel_sizes     = conv_params[1]
    self.stride_sizes     = conv_params[2]
    self.name             = name

    with tf.variable_scope(self.name):
      self.input    = tf.placeholder(tf.float32, [None, *state_size])
      self.action   = tf.placeholder(tf.float32, [None, action_size])
      self.Q_target = tf.placeholder(tf.float32, [None])

      if len(self.conv_filters) > 0:
        self.conv2d = tf.layers.conv2d(inputs             = self.input,
                                       filters            = self.conv_filters[0],
                                       kernel_size        = self.kernel_sizes[0],
                                       strides            = self.stride_sizes[0],
                                       activation         = tf.nn.relu,
                                       kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d()
                                      )

        self.output = tf.layers.dense(inputs             = tf.layers.flatten(self.conv2d),
                                      units              = self.action_size,
                                      kernel_initializer = tf.zeros_initializer()
                                     )
      else:
        self.output = tf.layers.dense(inputs             = tf.layers.flatten(self.input),
                                      units              = self.action_size,
                                      kernel_initializer = tf.zeros_initializer()
                                     )

      self.Q = tf.reduce_sum(tf.multiply(self.output, self.action), axis=1)
      self.loss = tf.losses.huber_loss(labels      = self.Q_target,
                                       predictions = self.Q)
      self.optimizer = tf.train.RMSPropOptimizer(learning_rate = self.optimizer_params[0],
                                                 momentum      = self.optimizer_params[1]).minimize(self.loss)
