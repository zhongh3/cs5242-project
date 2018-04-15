# Training Parameters
learning_rate = 1e-4

batch_size = 128

# total number of classes (0 or 1)
num_classes = 2

# dropout rate - the probability to drop a unit
dropout = 0.5

# sample length to read = in_height x in_width, the rest will be truncated
# e.g. 64 x 64 = 4096 bytes per sample
in_height = 64
in_width = 64

# data partition ratio
# train_ratio = 0.8 --> split data as 80% training, 20% testing
train_ratio = 0.8

# number of rows to read from *.csv
# set to None to read whole files
num_rows = 100

# number of epochs for training
train_epoch = 50
# number of steps for training, set to None to use train_epoch only
num_steps = 20
# when both train_epoch and num_steps are specified, the smaller number wins.

# maximum number of checkpoints to save, the earlier checkpoints will be overwritten
# please make sure that max_ckpt >= 1 to save the model trained
max_ckpt = 2
# interval to save checkpoints in terms of steps
# e.g. ckpt_steps = 100 --> save checkpoint every 100 steps
ckpt_steps = 10
