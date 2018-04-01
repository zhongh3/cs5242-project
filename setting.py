import os

# Training Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 128

# Network Parameters
num_input = 1024 # MNIST data input (img shape: 32*32)
num_classes = 2 # total classes (0 or 1)
dropout = 0.25 # Dropout, probability to drop a unit