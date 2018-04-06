import os

# Training Parameters
learning_rate = 1e-4
num_steps = None
batch_size = 128

# Network Parameters
num_classes = 2 # total classes (0 or 1)
dropout = 0.25 # Dropout, probability to drop a unit

train_ratio = 0.8
num_rows = 100
train_epoch = 5

max_ckpt = 160
ckpt_steps = 200