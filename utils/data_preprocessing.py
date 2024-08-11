# import numpy as np

# def read_csv(csv_path):
#     np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
#     path_XYs = []
#     for i in np.unique(np_path_XYs[:, 0]):
#         npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
#         XYs = []
#         for j in np.unique(npXYs[:, 0]):
#             XY = npXYs[npXYs[:, 0] == j][:, 1:]
#             XYs.append(XY)
#         path_XYs.append(XYs)
#     return path_XYs

# Example implementation of read_csv function
import pandas as pd

def read_csv(file_path):
    # data = read_csv('data/input/frag1.csv')
    df = pd.read_csv(file_path, header=None)
    data = df.values
    return data

# Example implementation of extract_features function
def extract_features(data):
    # Ensure the data is reshaped correctly
    return data.reshape(-1, 16, 16, 1).astype('float32')