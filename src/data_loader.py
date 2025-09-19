import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(data_dir='data', chunk_size=1000):
    """
    Loads data from text files, preprocesses it, and splits it into chunks.

    Args:
        data_dir (str): The path to the directory containing the data files.
        chunk_size (int): The size of each chunk to split the signals into.

    Returns:
        tuple: NumPy arrays for training and testing data (X_train, X_test, y_train, y_test).
    """
    col_names = [
        'Time', 'Left_VGRF1', 'Left_VGRF2', 'Left_VGRF3', 'Left_VGRF4',
        'Left_VGRF5', 'Left_VGRF6', 'Left_VGRF7', 'Left_VGRF8',
        'Right_VGRF1', 'Right_VGRF2', 'Right_VGRF3', 'Right_VGRF4',
        'Right_VGRF5', 'Right_VGRF6', 'Right_VGRF7', 'Right_VGRF8',
        'Force_Left', 'Force_Right'
    ]

    try:
        files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    except FileNotFoundError:
        print(f"Error: The directory '{data_dir}' was not found. Please place your data there.")
        return None, None, None, None

    data_frames = {}
    for file in files:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path, sep='\\t', header=None, names=col_names, encoding='utf-8', engine='python')
        label = 0 if 'co' in file.lower() else 1
        df['label'] = label
        data_frames[file] = df

    chunked_data_frames = {}
    for file, df in data_frames.items():
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        for i, chunk in enumerate(chunks):
            chunked_data_frames[f"{file}_chunk_{i + 1}"] = chunk

    filtered_data_frames = {k: v for k, v in chunked_data_frames.items() if v.shape == (chunk_size, 20)}

    result_array = np.array([df.values for df in filtered_data_frames.values()])

    left_minus_right = np.empty((len(result_array), chunk_size, 9), dtype=object)
    for i in range(len(result_array)):
        for j in range(1, 9):
            left_vgrf = result_array[i][:, j]
            right_vgrf = result_array[i][:, j + 8]
            left_minus_right[i][:, j - 1] = left_vgrf - right_vgrf
        left_minus_right[i][:, 8] = result_array[i][:, 19]

    X = left_minus_right[:, :, :8].astype('float32')
    y = left_minus_right[:, :, 8]
    y_final = np.array([label[0] for label in y]).astype('float32')

    mean_value = np.mean(X)
    std_value = np.std(X)
    X = (X - mean_value) / std_value

    X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=0.1, random_state=42, stratify=y_final)

    return X_train, X_test, y_train, y_test