# Image Tagger AI

An animal image classifier built with transfer learning on MobileNetV2. Select a photo and the app predicts which animal it is, with a confidence score and the top 3 candidates.

Trained on the [Animals-10 dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10) (butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel).

## Setup

```bash
pip install -r requirements.txt
```

## Training

Download the Animals-10 dataset and point the script at its folder (one subfolder per class):

```bash
python training.py --data-dir path/to/raw-img
```

Training runs in two phases:

1. **Head training**: the MobileNetV2 base is frozen and only the new classification layers are trained.
2. **Fine-tuning**: the top of the base model is unfrozen and trained with a very small learning rate (1e-5).

The script also:

- Uses MobileNetV2's own `preprocess_input` for both training and validation, matching prediction exactly
- Keeps augmentation out of the validation data
- Applies class weights so overrepresented classes (like dog) don't dominate underrepresented ones (like elephant)
- Saves the class label mapping to `class_labels.json` so prediction never relies on a hardcoded label order
- Saves the model as `animal_classifier_model.keras`

## Predicting

```bash
python animalTagger.py
```

Opens a simple GUI. Pick an image and it shows the prediction, the confidence, and the top 3 candidates. If the best confidence is below 60%, the app says it isn't sure instead of forcing a guess, since a photo of something outside the 10 known classes would otherwise still be labeled as an animal.

## Notes

- Models trained before these fixes used different preprocessing, so retrain before using `animalTagger.py`.
- `renamer.py` is a one-off helper used to normalize dataset file names.
