import os

# Training Parameters
learning_rate = 1e-4
num_steps = 10
batch_size = 128
train_epoch = 5

# Network Parameters
num_classes = 2 # total classes (0 or 1)
dropout = 0.25 # Dropout, probability to drop a unit

# sample length to read = in_height x in_width, the rest will be truncated
# e.g. 64 x 64 = 4096 bytes per sample
in_height = 96
in_width = 96

# train_ratio = 0.8 --> split data as 80% training, 20% testing
train_ratio = 0.8
num_rows = 100

max_ckpt = 160
ckpt_steps = 200
