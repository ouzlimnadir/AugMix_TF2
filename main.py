import os
import argparse
import logging
import numpy as np
import config
import cv2

entity = "medbaka74-university"
project = "DAv1"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from tensorflow.keras.utils import to_categorical
from trainer import train

import wandb

# extract the wandb api key from keys.txt
# with open("keys.txt", "r") as f:
#     api_key = f.readline().strip()

wandb.login()

###########################################################################

def get_cifar_data():
    """Loads agriculture crop images data"""
    dataset_path = os.path.join('data', 'crop_images')
    test_path = os.path.join('data', 'test_crop_images')

    # Initialize lists to store data
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # Function to load images from a directory
    def load_images_from_dir(directory):
        images = []
        labels = []
        label_map = {}  # To map label strings to numeric IDs

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):  # Assuming images are jpg or png
                    filepath = os.path.join(root, file)
                    label = os.path.basename(root)  # Label is the name of the folder
                    if label not in label_map:
                        label_map[label] = len(label_map)

                    # Read image using OpenCV
                    image = cv2.imread(filepath)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    image = image.astype('float32') / 255.0  # Normalize pixel values
                    # Resize image if needed
                    # image = cv2.resize(image, (desired_width, desired_height))  # Uncomment and adjust as needed
                    images.append(image)
                    labels.append(label_map[label])

        return images, labels

    # Load training data
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        if os.path.isdir(class_path):
            images, labels = load_images_from_dir(class_path)
            x_train.extend(images)
            y_train.extend(labels)

    # Load test data
    test_images, test_labels = load_images_from_dir(test_path)
    x_test.extend(test_images)
    y_test.extend(test_labels)

    # Convert lists to numpy arrays for easier manipulation (optional)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    y_train_cat = np.argmax(y_train, axis=1)
    y_test_cat = np.argmax(y_test, axis=1)

    return x_train, y_train_cat, x_test, y_test_cat, y_train, y_test





def parse_args():
    # Parse input arguments
    desc = "Implementation for AugMix paper"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--batch_size',
                        type=int,
                        required=False)
    parser.add_argument('--epochs',
                        help='number of training epochs',
                        type=int,
                        required=True)
    parser.add_argument('--max_lr',
                        help='maxium learning rate for lr scheduler',
                        default=1.0,
                        type=float)
    parser.add_argument('--min_lr',
                        help='minimum learning rate for lr scheduler',
                        default=1e-5,
                        type=float)
    parser.add_argument('--img_size',
                        help='size of the images',
                        default=32,
                        type=int)
    parser.add_argument("--save_dir_path",
                        type=str,
                        help="dir path to save output results",
                        default="",
                        required=False)
    parser.add_argument("--plot_name",
                        type=str,
                        help="filename for the plots",
                        default="history.png",
                        required=False)

    ## AUgMix params
    parser.add_argument("--jsd_loss",
                        type=bool,
                        help="To use jsd loss",
                        default=True,
                        required=False)
    parser.add_argument("--severity",
                        type=int,
                        help="Severity of underlying aug op",
                        default=3,
                        required=False)
    parser.add_argument("--width",
                        type=int,
                        help="Width of aug chain",
                        default=3,
                        required=False)
    parser.add_argument("--depth",
                        type=int,
                        help="Depth of aug chain. -1 or (1,3)",
                        default=-1,
                        required=False)
    parser.add_argument("--alpha",
                        type=float,
                        help="Probability coeff for distributions",
                        default=1.0,
                        required=False)
    args = vars(parser.parse_args())
    return args

###########################################################################

def main():
    args = parse_args()
    print('\nCalled with args:')
    for key in args:
        print(f"{key:<10}:  {args[key]}")
    print("="*78)

    # get the command line args
    config.max_lr = args["max_lr"]
    config.min_lr = args["min_lr"]
    config.batch_size = args["batch_size"]
    config.num_epochs = args["epochs"]
    config.IMAGE_SIZE = args["img_size"]
    config.plot_name = args["plot_name"]
    ## AugMix
    config.jsd_loss = args["jsd_loss"]
    config.severity = args["severity"]
    config.width = args["width"]
    config.depth = args["depth"]
    config.alpha = args["alpha"]

    
    if args["save_dir_path"] == "":
        config.save_dir_path = './model_checkpoints'
    else:
        config.save_dir_path = args["save_dir_path"]

    # get the data
    print("\nLoading data now.", end=" ")
    x_train, y_train, x_test, y_test, y_train_cat, y_test_cat = get_cifar_data()
    training_data = [x_train, y_train, y_train_cat]
    validation_data = [x_test, y_test, y_test_cat]
    print("Data loading complete. \n")

    # pass the arguments to the trainer
    train(training_data=training_data,
          validation_data=validation_data,
          cfg = config)
    

###########################################################################  


if __name__ == '__main__':

    wandb.init(entity=entity, project=project)
    main()
