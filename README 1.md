# Chest X-Ray Pneumonia Classifier

A CNN I built to look at chest X-ray images and figure out if the person has pneumonia or not.

## What This Is

I trained a CNN on chest X-ray images to classify them as either **NORMAL** or **PNEUMONIA**. My main goal was to catch as many pneumonia cases as possible — missing a sick patient is the worst outcome here.


## Dataset

- **Train:** 1,341 normal | 3,875 pneumonia
- **Val:** 8 normal | 8 pneumonia
- **Test:** 234 normal | 390 pneumonia

There's a big class imbalance — nearly 3x more pneumonia images than normal ones. I had to be careful about this or the model would just learn to always guess PNEUMONIA and still look accurate on paper.


## Methodology (Step by Step)

1. **Loaded the dataset** — I mounted Google Drive, extracted the zip, and set up folder paths for train/val/test.
2. **Handled class imbalance** — I computed class weights so the model pays more attention to the minority class (NORMAL).
3. **Preprocessed & augmented images** — I rescaled all images to [0,1] and applied light augmentation on training data only.
4. **Built the CNN** — 3 convolutional blocks with increasing filters (32→64→128), followed by dense layers.
5. **Compiled the model** — I used Adam optimizer, binary crossentropy loss, and tracked accuracy, AUC, precision, and recall.
6. **Set up callbacks** — early stopping, model checkpointing based on val_auc, and learning rate reduction when val_loss plateaus.
7. **Trained the model** — I ran it for up to 4 epochs with class weights applied.
8. **Evaluated on test set** — I checked accuracy, precision, recall, and F1 score.


## Approach

### Handling Class Imbalance
I used `compute_class_weight` from sklearn to balance the training data. This gave NORMAL images more weight during training, which helped prevent the model from taking the lazy route of always predicting PNEUMONIA.

### Data Augmentation
I kept augmentation light since X-rays are clinical images — too much distortion makes them unrealistic:
- Small rotations (±15°)
- Slight zoom (20%) and shear (10%)
- Horizontal flipping

I only applied these to the training data. Val and test sets just get rescaled.

### Model Architecture
- 3 conv blocks with filters going 32 → 64 → 128
- Each block has two conv layers, BatchNorm, LeakyReLU, MaxPooling, and Dropout (25%)
- I used **LeakyReLU** instead of regular ReLU — ReLU kills neurons once they go negative, LeakyReLU keeps a small gradient flowing so they don't just die
- **BatchNorm** after every conv layer to keep activations stable and training smooth
- **He Normal** initialization throughout, since that's what works best with ReLU-family activations
- Three dense layers (512 → 256 → 128) with Dropout (50%, 50%, 40%)
- Output: single neuron with sigmoid, since this is binary classification

### Training Setup
- Optimizer: Adam (lr=0.001)
- Loss: Binary crossentropy
- I monitored **val_auc** for saving the best model — it's a better signal than accuracy for imbalanced data
- `ReduceLROnPlateau` halves the learning rate if val_loss doesn't improve for 3 epochs
- `EarlyStopping` kicks in after 5 epochs of no improvement and restores the best weights


## Libraries I Used

- **TensorFlow / Keras** — main framework for building and training the CNN
- **NumPy** — array operations and handling class labels
- **Matplotlib** — plotting sample images and training curves
- **Scikit-learn** — computing class weights and evaluation metrics


## Findings

| Metric    | Score  |
|-----------|--------|
| Accuracy  | —      |
| Precision | —      |
| Recall    | —      |
| F1 Score  | —      |
| AUC       | —      |

*(Fill in after running evaluation on the test set)*

The model is optimized to prioritize recall — it's better to flag a healthy patient for a follow-up than to miss someone who's actually sick. The AUC metric during training gives a clearer picture of how well the model is actually discriminating between the two classes, which is why I used it for checkpointing.


## Limitations

- The model was only run for up to 4 epochs, so there's room to improve with longer training.
- The architecture is built from scratch — using a pretrained backbone like ResNet or EfficientNet would likely give better results with less training time.
- Threshold tuning wasn't explored here. The default 0.5 cutoff might not be optimal — lowering it would boost recall at the cost of precision.
