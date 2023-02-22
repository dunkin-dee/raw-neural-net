import numpy as np

softmax_outputs = np.array([[0.0, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]])

class_targets2 = np.argmax(class_targets, axis=1)

print(-np.log(np.clip(softmax_outputs, 1e-10, 1-1e-10)[range(len(softmax_outputs)), [np.argmax(class_targets, axis=1)]]))
