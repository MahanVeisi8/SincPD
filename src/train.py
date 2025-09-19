import tensorflow as tf
from data_loader import load_and_preprocess_data
from model import build_sincnet_model
from visualization import plot_training_history
import os

def main():
    """
    Main function to train the SincNet model.
    """
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_dir='data')

    if X_train is None:
        return

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    print("Building SincNet model...")
    model = build_sincnet_model(input_shape=(1000, 8))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=200,
        restore_best_weights=True
    )

    print("Starting model training...")
    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=500,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    if not os.path.exists('sincnet_model'):
        os.makedirs('sincnet_model')
    model.save("sincnet_model/sincnet_parkinson_model.h5")
    print("Trained model saved successfully in the 'sincnet_model' directory.")

    if not os.path.exists('results'):
        os.makedirs('results')
    plot_training_history(history, save_path='results/training_plots.pdf')
    print("Training plots saved in the 'results' directory.")

    print("Evaluating model on test data...")
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")

if __name__ == '__main__':
    main()