# cython: boundscheck=False, wraparound=False, initializedcheck=False
from libc.math cimport fmaf
import numpy as np

cdef extern from * nogil:
    """
    float xexpf(float);
    float xlogf_u1(float);
    """
    float xexpf(float)
    float xlogf_u1(float)

def forward(float[:, ::1] inputs, float[:, ::1] weights, float[::1] bias):
    if inputs.shape[1] != weights.shape[1] or weights.shape[0] != bias.shape[0]:
        raise ValueError('Linear dimensions do not match')
    result = np.empty((inputs.shape[0], weights.shape[0]), dtype=np.float32)
    cdef float[:, ::1] output = result
    cdef Py_ssize_t row, column, k
    cdef float value
    with nogil:
        for row in range(inputs.shape[0]):
            for column in range(weights.shape[0]):
                value = bias[column]
                for k in range(inputs.shape[1]):
                    value = fmaf(inputs[row, k], weights[column, k], value)
                output[row, column] = value
    return result

def log_softmax(float[:, ::1] inputs):
    result = np.empty((inputs.shape[0], inputs.shape[1]), dtype=np.float32)
    cdef float[:, ::1] output = result
    cdef Py_ssize_t row, col, lane
    cdef float maximum, total, logtotal
    cdef float sums[4]
    if inputs.shape[1] < 4:
        raise ValueError('This isolated log-softmax candidate requires at least four classes')
    with nogil:
        for row in range(inputs.shape[0]):
            maximum = inputs[row, 0]
            for col in range(1, inputs.shape[1]):
                if inputs[row,col] > maximum: maximum = inputs[row,col]
            for lane in range(4): sums[lane] = xexpf(inputs[row,lane] - maximum)
            for col in range(4, inputs.shape[1]):
                lane = col % 4
                sums[lane] = sums[lane] + xexpf(inputs[row,col] - maximum)
            total = (sums[0] + sums[2]) + (sums[1] + sums[3])
            logtotal = xlogf_u1(total)
            for col in range(inputs.shape[1]):
                output[row,col] = (inputs[row,col] - maximum) - logtotal
    return result

def cross_entropy(float[:, ::1] inputs, long long[::1] targets):
    if inputs.shape[0] != targets.shape[0] or not targets.shape[0]:
        raise ValueError('Invalid target dimensions')
    for target in targets:
        if target < 0 or target >= inputs.shape[1]: raise ValueError('Invalid target label')
    logp_array = log_softmax(inputs)
    gradient_array = np.empty_like(logp_array)
    cdef float[:,::1] logp = logp_array
    cdef float[:,::1] gradient = gradient_array
    cdef float partial[8]
    cdef float inv = -1.0 / <float>inputs.shape[0]
    cdef float value, loss = 0
    cdef Py_ssize_t row, col, level, level_power = max(4, (inputs.shape[0]-1).bit_length() // 8)
    cdef unsigned long long mask, base_mask = (1 << level_power) - 1
    for level in range(8): partial[level] = 0
    with nogil:
        for row in range(inputs.shape[0]):
            partial[0] = partial[0] - logp[row, targets[row]]
            for level in range(7):
                mask = base_mask << (level * level_power)
                if row & mask: break
                partial[level+1] = partial[level+1] + partial[level]
                partial[level] = 0
            for col in range(inputs.shape[1]):
                value = inv if col == targets[row] else 0
                gradient[row,col] = value - (xexpf(logp[row,col]) * inv)
        for level in range(8): loss = loss + partial[level]
        loss = loss / <float>inputs.shape[0]
    return np.float32(loss), gradient_array

def bias_gradient(float[:, ::1] gradient):
    result = np.empty(gradient.shape[1], dtype=np.float32)
    cdef float[::1] output = result
    cdef float acc[4][4]
    cdef Py_ssize_t col, level, lane, i, j, size = gradient.shape[0] // 4
    cdef Py_ssize_t power = max(4, (max(size,1)-1).bit_length() // 4)
    cdef Py_ssize_t step = 1 << power, mask = step - 1
    with nogil:
        for col in range(gradient.shape[1]):
            for level in range(4):
                for lane in range(4): acc[level][lane] = 0
            i = 0
            while i + step <= size:
                for j in range(step):
                    for lane in range(4): acc[0][lane] = acc[0][lane] + gradient[i*4+lane,col]
                    i = i + 1
                for level in range(1,4):
                    for lane in range(4):
                        acc[level][lane] = acc[level][lane] + acc[level-1][lane]
                        acc[level-1][lane] = 0
                    if i & (mask << (level*power)): break
            while i < size:
                for lane in range(4): acc[0][lane] = acc[0][lane] + gradient[i*4+lane,col]
                i = i + 1
            for level in range(1,4):
                for lane in range(4): acc[0][lane] = acc[0][lane] + acc[level][lane]
            for i in range(size*4, gradient.shape[0]): acc[0][0] = acc[0][0] + gradient[i,col]
            for lane in range(1,4): acc[0][0] = acc[0][0] + acc[0][lane]
            output[col] = acc[0][0]
    return result

def scaled_add(float[::1] values, float[::1] increments, float alpha):
    if values.shape[0] != increments.shape[0]: raise ValueError('Update shapes differ')
    result = np.empty(values.shape[0], dtype=np.float32)
    cdef float[::1] output = result
    cdef Py_ssize_t i
    with nogil:
        for i in range(values.shape[0]): output[i] = fmaf(alpha, increments[i], values[i])
    return result
