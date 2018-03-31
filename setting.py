import os

batch_size = 128
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20


save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'