import json
import argparse

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Parse the dataset path from the command line so no machine-specific
# paths are hardcoded in the script.
parser = argparse.ArgumentParser(description="Train the animal classifier.")
parser.add_argument(
    "--data-dir",
    default="raw-img",
    help="Path to the dataset folder (one subfolder per class)",
)
args = parser.parse_args()

rootFolder = args.data_dir
imageSize = (224, 224)
batchSize = 32

# Phase 1 trains only the new classification head on top of the frozen
# base model. Phase 2 unfreezes the top of the base model and fine-tunes
# it with a much smaller learning rate.
headEpochs = 10
fineTuneEpochs = 5
fineTuneFromLayer = 100

# Both generators use MobileNetV2's own preprocess_input (which scales
# pixels to the -1..1 range) so training matches prediction exactly.
# Augmentation is applied to the training data only.
trainDataGenerator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

# The validation generator only preprocesses. Validating on augmented
# images gives a noisy, misleading accuracy number.
validationDataGenerator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
)

trainGenerator = trainDataGenerator.flow_from_directory(
    rootFolder,
    target_size=imageSize,
    batch_size=batchSize,
    class_mode="categorical",
    subset="training",
    seed=42,
)

validationGenerator = validationDataGenerator.flow_from_directory(
    rootFolder,
    target_size=imageSize,
    batch_size=batchSize,
    class_mode="categorical",
    subset="validation",
    seed=42,
    shuffle=False,
)

# Save the class index mapping so prediction code always uses the exact
# label order the model was trained with, instead of a hardcoded list.
classIndices = trainGenerator.class_indices
indexToLabel = {index: label for label, index in classIndices.items()}
with open("class_labels.json", "w") as labelFile:
    json.dump(indexToLabel, labelFile, indent=2)
print(f"Saved class label mapping: {indexToLabel}")

# Weight each class by its inverse frequency so common classes (like dog)
# don't dominate training over rare ones (like elephant).
labels = trainGenerator.classes
totalSamples = len(labels)
numClasses = trainGenerator.num_classes
classWeights = {}
for classIndex in range(numClasses):
    classCount = int((labels == classIndex).sum())
    classWeights[classIndex] = totalSamples / (numClasses * classCount)
print(f"Class weights: {classWeights}")

# Load the pre-trained MobileNetV2 base without its top layer, and freeze
# it so phase 1 trains only the new head.
baseModel = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
baseModel.trainable = False

x = baseModel.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
predictions = Dense(numClasses, activation="softmax")(x)

model = Model(inputs=baseModel.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

earlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=3, restore_best_weights=True
)

print("Phase 1: training the classification head (base model frozen)")
model.fit(
    trainGenerator,
    validation_data=validationGenerator,
    epochs=headEpochs,
    class_weight=classWeights,
    callbacks=[earlyStopping],
)

# Phase 2: unfreeze the top of the base model and fine-tune it with a
# very small learning rate so the pretrained features aren't destroyed.
baseModel.trainable = True
for layer in baseModel.layers[:fineTuneFromLayer]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("Phase 2: fine-tuning the top of the base model")
model.fit(
    trainGenerator,
    validation_data=validationGenerator,
    epochs=fineTuneEpochs,
    class_weight=classWeights,
    callbacks=[earlyStopping],
)

# Save in the modern Keras format (.h5 is legacy).
model.save("animal_classifier_model.keras")
print("Saved model to animal_classifier_model.keras")
