# import matplotlib.pyplot as plt

# def plot(paths_XYs):
#     fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
#     for i, XYs in enumerate(paths_XYs):
#         c = 'C{}'.format(i % 10)
#         for XY in XYs:
#             ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
#     ax.set_aspect('equal')
#     plt.show()


import matplotlib.pyplot as plt

def plot_predictions(images, predictions, true_labels, class_names):
    fig, axes = plt.subplots(nrows=1, ncols=len(images), figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f"Pred: {class_names[predictions[i]]}\nTrue: {class_names[true_labels[i]]}")
        ax.axis('off')
    plt.show()