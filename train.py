import argparse,os
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # uncomment this if you want to run in CPU
from model_size import get_model_memory_usage
from datetime import datetime
from keras import optimizers
from keras.callbacks import ModelCheckpoint,TensorBoard
import Utils



# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str)
parser.add_argument("--train_images", type = str)
parser.add_argument("--train_annotations", type = str)
parser.add_argument("--logs", type = str, default = "")
parser.add_argument("--n_classes", type=int )
parser.add_argument("--input_height", type=int , default = 640  )
parser.add_argument("--input_width", type=int , default = 640 )

parser.add_argument("--val_images", type = str , default = "")
parser.add_argument("--val_annotations", type = str , default = "")

parser.add_argument("--start_epoch", type = int, default = 0)
parser.add_argument("--end_epoch", type = int, default = 10)
parser.add_argument("--batch_size", type = int, default = 1)
parser.add_argument("--val_batch_size", type = int, default = 1)
parser.add_argument("--load_weights", type = str , default = "")

parser.add_argument("--optimizer_name", type = str , default = "")
parser.add_argument("--init_learning_rate", type = str , default = 0.001)
parser.add_argument("--epoch_steps", type = str , default = 200)



# Assign command line arguments to parameter
args = parser.parse_args()

train_images_path = args.train_images
train_labels_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
logs_dir = args.logs

save_weights_path = args.save_weights_path
start_epoch = args.start_epoch
end_epoch = args.end_epoch
trained_weights = args.load_weights

optimizer_name = args.optimizer_name
learning_rate = float(args.init_learning_rate)
steps = int(args.epoch_steps)

val_images_path = args.val_images
val_labels_path = args.val_annotations
val_batch_size = args.val_batch_size


# 1. Initialize a model and Load the weights in case you continue a previous training
if len(trained_weights) > 0:
    keras_model = Utils.VGG16_Unet(n_classes, False, input_height=input_height, input_width=input_width)
    keras_model.load_weights(trained_weights)
else:
    keras_model = Utils.VGG16_Unet(n_classes, True, input_height=input_height, input_width=input_width)


# 2. Define the output size of your results.
output_height = keras_model.outputHeight
output_width = keras_model.outputWidth
print("Model output shape", keras_model.output_shape)

# Use this lines if you like to determine the memory size of the model and see the model architecture
# model_mem_size = get_model_memory_usage(train_batch_size, keras_model)
# print("Model memory size is: {}GB".format(model_mem_size))
# keras_model.summary()


# 3. Load you batches for the train and validation dataset
train_data = Utils.image_labels_generator(train_images_path, train_labels_path, train_batch_size, n_classes, input_height, input_width, output_height, output_width)

val_data = Utils.image_labels_generator(val_labels_path, val_labels_path, val_batch_size, n_classes, input_height, input_width, output_height, output_width)



# 4. Select the optimizer and the learning rate (default option is Adam)
if optimizer_name == 'rmsprop':
    optimizer = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
elif optimizer_name == 'adadelta':
    optimizer = optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=None, decay=0.0)
elif optimizer_name == 'sgd':
    optimizer = optimizers.SGD(lr=learning_rate, decay=0.0)
else:
    optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


# 5. Set the callbacks for saving the weights and the tensorboard
weights = save_weights_path + "_lr_{}".format(round(learning_rate,8)) + "_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(weights, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True)
logs_dir = logs_dir + "/lr_{}".format(round(learning_rate,8))
tensorboard = TensorBoard(log_dir=logs_dir, histogram_freq=0, write_images=False)


# 6. Compile the model with the selected optimizer
keras_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# 7. Train your network
keras_model.fit_generator(train_data, steps, validation_data=val_data, validation_steps=100, epochs=end_epoch, callbacks=[checkpoint,tensorboard], initial_epoch=start_epoch)
