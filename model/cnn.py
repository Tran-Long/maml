from flax import linen as nn  # Linen API


    
class CNNModuleWithBatchnorm(nn.Module):
    """A simple CNN model with batch normalization."""

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        return x
    
class CNNModule(nn.Module):
    """A simple CNN model without batch normalization."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        return x

class CNN(nn.Module):
    """A simple CNN model."""
    n_modules: int = 4
    n_classes: int = 5

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_modules):
            x = CNNModule()(x)
        x = nn.avg_pool(x, window_shape=(x.shape[1], x.shape[2]))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_classes)(x)
        return x
  
class CNNWithBatchNorm(nn.Module):
    """A simple CNN model."""
    n_modules: int = 4

    @nn.compact
    def __call__(self, x, train):
        for _ in range(self.n_modules):
            x = CNNModule()(x, train=train)
        x = nn.avg_pool(x, window_shape=(x.shape[1], x.shape[2]))
        x = x.reshape((x.shape[0], -1))
        return x