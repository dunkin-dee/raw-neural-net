# """
# solving for x in

# e**x = b
# x = ln(b)
# """

# import numpy as np
# import math
# b = 5.2

# print(np.log(b))

# print(math.e ** 1.6486586255873816)

import math
import numpy as np

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = -sum([target*math.log(pred) for target, pred in zip(target_output, softmax_output)])
loss1 = -(np.log(softmax_output[np.argmax(target_output)]))


print(loss)
print(loss1)