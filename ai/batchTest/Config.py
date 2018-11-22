# Esta classe conterá os hiper parâmetros relevantes a DQN

class Config:
  def __init__(self, conv_filters, kernel_sizes, stride_sizes, units, learning_rate, momentum, epsilon):
    # Camadas ocultas
    self.conv_filters = conv_filters
    self.kernel_sizes = kernel_sizes
    self.stride_sizes = stride_sizes
    self.units = units

    # Cálculo de erro
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.epsilon = epsilon

