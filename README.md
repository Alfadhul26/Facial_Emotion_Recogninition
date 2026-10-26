# Facial Emotion Recognition

A small neural network that classifies a 48×48 grayscale face image into one of 7 emotions, with Streamlit deployment that takes a webcam photo and predicts the emotion.

**🔗 Live demo:** <LINK>

## Emotions

`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

## Project structure

```
├── facial_emotional_recognition.ipynb   # data loading, training, evaluation
├── app.py                               # Streamlit Link
├── emotion_model.keras                  # trained model
└── requirements.txt
```

## Data

- 28,709 training images and 7,178 test images, 48×48 grayscale, 7 classes (loaded with `image_dataset_from_directory`).
- Training images were split 80/20 into train (22,968) and validation (5,741) with `seed=42`. The test set was used only for the final evaluation.
- Classes are imbalanced: `happy` has ~5,800 training images, while `disgust` has only ~360.
- Source: Kaggle <https://www.kaggle.com/datasets/msambare/fer2013/data>

## Model architecture

A plain fully connected network (no convolutions):

```
Input (48×48) → Rescaling(1/255) → Flatten
  → Dense(512, ReLU) → Dense(256, ReLU) → Dense(100, ReLU) → Dense(50, ReLU)
  → Dense(7, softmax)
```

Choices:
- **Rescaling inside the model**
- **Class weights** (`balanced`) to counter the imbalance.
- **Adam, sparse categorical cross-entropy**
- A first baseline with default settings and no class weights reached ~36% validation accuracy. The class-weighted model was then trained for up to 50 epochs.

## Performance (test set, 7,178 images)

| Emotion  | Precision | Recall | F1   | Support |
|----------|-----------|--------|------|---------|
| angry    | 0.21      | 0.22   | 0.22 | 958     |
| disgust  | 0.08      | 0.43   | 0.14 | 111     |
| fear     | 0.18      | 0.21   | 0.19 | 1024    |
| happy    | 0.58      | 0.38   | 0.46 | 1774    |
| neutral  | 0.33      | 0.31   | 0.32 | 1233    |
| sad      | 0.29      | 0.28   | 0.29 | 1247    |
| surprise | 0.45      | 0.49   | 0.47 | 831     |
| **Accuracy** |       |        | **0.32** | 7178 |
| Macro avg | 0.30     | 0.33   | 0.30 |         |

**Next steps:** CNN and data augmentation.

## limitations

This model is a learning project and **should not be used to make decisions about people** (hiring, education, security, mental health, etc.).

Where it is likely to fail:

- **Low accuracy overall.** About two out of three predictions on the test set are wrong. `fear`, `angry` and `disgust` are the weakest classes.
- **Lighting and image quality.** It was trained on small, low-resolution grayscale faces. Dim, backlit, strongly shadowed or blurry webcam images will degrade results.
- **Class imbalance.** `disgust` has very few examples, so predictions for it are unreliable; `happy` dominates the training data.
- **Privacy.** Webcam images are processed in the app session only; if you deploy your own copy, do not store user images without consent.
