import math

#softmax_output_example = [0.7, 0.1, 0.2]
#target_output = [1, 0, 0]

#softmax_output_example = [0.22, 0.6, 0.18]
#target_output = [0, 1, 0]

softmax_output_example = [0.32, 0.36, 0.32]
target_output = [0, 1, 0]

# hard coded loss function
loss = -(math.log(softmax_output_example[0]) * target_output[0] +
         math.log(softmax_output_example[1]) * target_output[1] +
         math.log(softmax_output_example[2]) * target_output[2])

#print(loss)

# knowing that our target is 1, 0, 0
loss = -(math.log(softmax_output_example[1]))

#print(loss)

# LOSS gets larger the lower our confidence is

print(math.log(1.))
print(math.log(0.95))
print(math.log(0.9))
print(math.log(0.8))
print('...')
print(math.log(0.2))
print(math.log(0.1))
print(math.log(0.05))
print(math.log(0.01))

import numpy as np
print('\n\n\n\n\n')

b = 5.2
print(np.log(b))
print(math.e ** 1.6486586255873816) 