"""
Shared neural network model definitions.

All optimizer implementations live in ``optimisers/`` and are registered in
``optimizer_registry.py``. This module only contains Flax model classes.
"""

import jax.numpy as jnp
from flax import linen as nn


class CNN(nn.Module):
    """CNN model that can be used for both CIFAR-10 and MNIST"""
    num_classes: int = 10

    @nn.compact
    def __call__(self, x):
        # Handle different input shapes
        if len(x.shape) == 2:  # MNIST case (flattened)
            x = x.reshape((x.shape[0], 28, 28, 1))
        elif x.shape[-1] == 3:  # CIFAR-10 case (already 32x32x3)
            pass
        else:  # MNIST case (already reshaped)
            pass

        # First convolutional block
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.gelu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Second convolutional block
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.gelu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Third convolutional block (optional for more complex tasks)
        if x.shape[1] > 4:  # Only add if spatial dimensions allow
            x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.gelu(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
            dense_features = 256
        else:
            dense_features = 128

        # Flatten and dense layers
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=dense_features)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x


class MLP(nn.Module):
    """Multi-layer perceptron for regression and classification tasks"""
    features: int = 64
    output_dim: int = 10
    activation: nn.activation = nn.gelu

    def setup(self):
        self.dense1 = nn.Dense(self.features)
        self.dense2 = nn.Dense(self.features)
        self.dense3 = nn.Dense(self.output_dim)

    def __call__(self, x):
        x = self.activation(self.dense1(x))
        x = self.activation(self.dense2(x))
        x = self.dense3(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for ResNet architecture"""
    features: int
    strides: tuple = (1, 1)

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x

        # First convolution
        y = nn.Conv(features=self.features, kernel_size=(3, 3),
                    strides=self.strides, padding='SAME')(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)

        # Second convolution
        y = nn.Conv(features=self.features, kernel_size=(3, 3),
                    strides=(1, 1), padding='SAME')(y)
        y = nn.BatchNorm(use_running_average=not train)(y)

        # Projection shortcut if dimensions change
        if x.shape != y.shape:
            residual = nn.Conv(features=self.features, kernel_size=(1, 1),
                              strides=self.strides, padding='SAME')(x)
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        # Add residual connection and apply ReLU
        y = y + residual
        return nn.relu(y)


class ResNet18(nn.Module):
    """ResNet-18 architecture optimized for CIFAR-10"""
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Handle different input shapes
        if len(x.shape) == 2:  # Flattened input
            x = x.reshape((x.shape[0], 28, 28, 1))

        # Initial convolution (smaller for CIFAR-10)
        x = nn.Conv(features=64, kernel_size=(3, 3),
                   strides=(1, 1), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # Layer 1 (2 residual blocks, no downsampling)
        x = ResidualBlock(features=64, strides=(1, 1))(x, train=train)
        x = ResidualBlock(features=64, strides=(1, 1))(x, train=train)

        # Layer 2 (2 residual blocks, downsample)
        x = ResidualBlock(features=128, strides=(2, 2))(x, train=train)
        x = ResidualBlock(features=128, strides=(1, 1))(x, train=train)

        # Layer 3 (2 residual blocks, downsample)
        x = ResidualBlock(features=256, strides=(2, 2))(x, train=train)
        x = ResidualBlock(features=256, strides=(1, 1))(x, train=train)

        # Layer 4 (2 residual blocks, downsample)
        x = ResidualBlock(features=512, strides=(2, 2))(x, train=train)
        x = ResidualBlock(features=512, strides=(1, 1))(x, train=train)

        # Global average pooling and final dense layer
        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        x = nn.Dense(features=self.num_classes)(x)

        return x


class MiniGPT(nn.Module):
    """Small GPT model for character-level language modeling on tiny Shakespeare"""
    vocab_size: int  # Size of vocabulary (unique characters)
    embed_dim: int = 128  # Embedding dimension
    num_heads: int = 4  # Number of attention heads
    num_layers: int = 6  # Number of transformer layers
    dropout_rate: float = 0.1  # Dropout rate
    max_seq_len: int = 256  # Maximum sequence length

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Create positional embeddings
        batch_size, seq_len = x.shape
        positions = jnp.arange(seq_len)[None, :]  # [1, seq_len]

        # Token and position embeddings
        token_embed = nn.Embed(self.vocab_size, self.embed_dim)(x)
        pos_embed = nn.Embed(self.max_seq_len, self.embed_dim)(positions)
        x = token_embed + pos_embed

        # Apply dropout to embeddings
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        # Create causal mask for autoregressive attention
        causal_mask = nn.make_causal_mask(jnp.ones((batch_size, seq_len)))

        # Transformer layers
        for _ in range(self.num_layers):
            # Attention block with residual connection and layer norm
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.SelfAttention(
                num_heads=self.num_heads,
                qkv_features=self.embed_dim,
                dropout_rate=self.dropout_rate,
                deterministic=not train
            )(x, mask=causal_mask)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
            x = x + residual

            # MLP block with residual connection and layer norm
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.Dense(features=4 * self.embed_dim)(x)
            x = nn.gelu(x)
            x = nn.Dense(features=self.embed_dim)(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
            x = x + residual

        # Final layer norm and projection to vocabulary
        x = nn.LayerNorm()(x)
        logits = nn.Dense(features=self.vocab_size)(x)

        return logits
