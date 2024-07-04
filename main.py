import os
import argparse
import logging
import numpy as np
import config
import cv2

entity = "um6p2024"
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
    dataset_path = os.path.join('data', 'crop_images')
    test_path = os.path.join('data', 'test_crop_image')

    # Initialisation des listes pour stocker les données
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # Chargement des images d'entraînement
    labels_set = set(os.listdir(dataset_path))

    for label in labels_set:
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                image = cv2.imread(image_path)  # Charger l'image avec OpenCV
                if image is not None:
                    x_train.append(image)
                    y_train.append(label)

    # Chargement des images de test
    for image_name in os.listdir(test_path):
        image_path = os.path.join(test_path, image_name)
        image = cv2.imread(image_path)  # Charger l'image avec OpenCV
        if image is not None:
            # Utilisation du nom de l'image comme label
            label = image_name.split('.')[0]  # Enlever l'extension
            if label in labels_set:
                x_test.append(image)
                y_test.append(label)
            else:
                print(f"Attention: Le label '{label}' pour l'image '{image_name}' not in training folder.")

    # Conversion des listes en tableaux numpy
    x_train = np.array(x_train)
    y_train_cat = np.array(y_train)
    x_test = np.array(x_test)
    y_test_cat = np.array(y_test)

    print(f"Loaded {x_train.shape} training images and {x_test.shape} test images.")
    print(f"Loaded {y_train_cat.shape} unique classes.")
    print(f"Loaded {y_test_cat.shape} unique classes.")

    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    print(f"y_train_cat shape: {y_train.shape}")
    print(f"y_test_cat shape: {y_test.shape}")

    return x_train, y_train, x_test, y_test, y_train_cat, y_test_cat


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
