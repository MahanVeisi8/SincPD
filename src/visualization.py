import matplotlib.pyplot as plt

def plot_training_history(history, save_path=None):
    """
    Plots and saves the training history graphs.

    Args:
        history (tf.keras.callbacks.History): The training history object.
        save_path (str, optional): Path to save the plot.
    """
    history_dict = history.history
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 10))

    # Accuracy Plot
    plt.subplot(2, 2, 1)
    plt.plot(history_dict['accuracy'], label='Training Accuracy', linestyle='--', color='black')
    plt.plot(history_dict['val_accuracy'], label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(2, 2, 2)
    plt.plot(history_dict['loss'], label='Training Loss', linestyle='--', color='black')
    plt.plot(history_dict['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Precision Plot
    plt.subplot(2, 2, 3)
    plt.plot(history_dict['precision'], label='Training Precision', linestyle='--', color='black')
    plt.plot(history_dict['val_precision'], label='Validation Precision', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Training and Validation Precision')
    plt.legend()

    # Recall Plot
    plt.subplot(2, 2, 4)
    plt.plot(history_dict['recall'], label='Training Recall', linestyle='--', color='black')
    plt.plot(history_dict['val_recall'], label='Validation Recall', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Training and Validation Recall')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()