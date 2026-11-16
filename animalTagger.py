import json
import os

import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tkinter import Tk, Label, Button, filedialog, Text, Canvas

MODEL_PATH = "animal_classifier_model.keras"
LEGACY_MODEL_PATH = "animal_classifier_model.h5"
LABELS_PATH = "class_labels.json"

# Below this confidence the app says it isn't sure instead of forcing a
# guess. Softmax always picks one of the known classes, so without a
# threshold a photo of a car would still come back as some animal.
CONFIDENCE_THRESHOLD = 0.60
TOP_K = 3

# Load the trained model, falling back to the legacy .h5 file if the
# newer .keras file isn't present.
modelPath = MODEL_PATH if os.path.exists(MODEL_PATH) else LEGACY_MODEL_PATH
model = load_model(modelPath)

# Load the label order saved by training.py so predictions always use
# the exact class order the model was trained with.
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(
        f"{LABELS_PATH} not found. Run training.py first; it saves the "
        "class label mapping alongside the model."
    )
with open(LABELS_PATH) as labelFile:
    indexToLabel = {int(index): label for index, label in json.load(labelFile).items()}


def predict_image(imagePath, model):
    """Return the top predictions for an image as (label, probability) pairs."""
    img = load_img(imagePath, target_size=(224, 224))
    imgArray = img_to_array(img)
    imgArray = np.expand_dims(imgArray, axis=0)
    # Must match the preprocessing used in training.py exactly.
    imgArray = preprocess_input(imgArray)

    predictions = model.predict(imgArray)[0]
    topIndices = np.argsort(predictions)[::-1][:TOP_K]
    return [(indexToLabel[index], float(predictions[index])) for index in topIndices]


class ImageClassifierGUI:
    def __init__(self, master):
        self.master = master
        master.geometry("800x600")
        master.title("Image Classifier")

        knownAnimals = ", ".join(
            sorted(label.capitalize() for label in indexToLabel.values())
        )
        self.label = Label(
            master,
            text=f"Select an image to classify. Known animals: {knownAnimals}.",
        )
        self.label.pack()

        self.selectButton = Button(
            master, text="Select Image", command=self.select_image
        )
        self.selectButton.pack()

        self.canvas = Canvas(master, width=400, height=400)
        self.canvas.pack()

        self.classificationText = Text(master, height=4, wrap="none")
        self.classificationText.pack()

    def select_image(self):
        filePath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
        )
        if not filePath:
            return

        topPredictions = predict_image(filePath, model)
        bestLabel, bestConfidence = topPredictions[0]

        self.classificationText.delete(1.0, "end")
        if bestConfidence < CONFIDENCE_THRESHOLD:
            self.classificationText.insert(
                "end",
                "Not confident this is one of the known animals.\n",
            )
        else:
            self.classificationText.insert(
                "end",
                f"Classification: {bestLabel.capitalize()} "
                f"({bestConfidence * 100:.2f}% confidence)\n",
            )
        for label, confidence in topPredictions:
            self.classificationText.insert(
                "end", f"  {label.capitalize()}: {confidence * 100:.2f}%\n"
            )

        image = Image.open(filePath)
        image.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=photo, anchor="nw")
        self.canvas.image = photo


if __name__ == "__main__":
    root = Tk()
    gui = ImageClassifierGUI(root)
    root.mainloop()
