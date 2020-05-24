import numpy as np

def sigmoid(x):
	return 1/(1 + np.exp(-x)) 

w_proj = np.array([[-0.0484, -0.0198, -0.2165],
[ 0.1328, -0.3303, -0.1018],
[ 0.2238,  0.5419,  0.1360]]).transpose()
w_gate = np.array([[ 0.2357,  0.0661,  0.2262],
[ 0.5599, -0.2397, -0.0204],
[ 0.1328, -0.0038, -0.0553]]).transpose()
b_proj = np.array([-0.1959,  0.0554, -0.0647])
b_gate = np.array([-0.5109, -0.4980, -0.5195])
input = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
expected_output = [[0.0421, 0.0738, 0.3570],
        [0.1648, 0.1709, 0.3601]]

x_proj = input.dot(w_proj) + b_proj
x_gate = input.dot(w_gate) + b_gate
s_gate = sigmoid(x_gate)
print(s_gate * x_proj + (1-s_gate) * input)
