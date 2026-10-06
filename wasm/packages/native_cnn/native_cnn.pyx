# cython: boundscheck=False, wraparound=False, initializedcheck=False
import numpy as np

cdef extern from * nogil:
    """
    int course_nnpack_fft16(const float*,const float*,const float*,float*,int,int,int,int,int);
    void course_conv_forward(const float*,const float*,const float*,float*,int,int,int,int,int,int,int,int,int,int,int);
    void course_linear_split(const float*,const float*,const float*,float*,int,int,int);
    void course_bias_outer(const float*,float*,int,int);
    void course_conv_bias(const float*,float*,int,int,int,int);
    void course_conv_weight(const float*,const float*,float*,int,int,int,int,int);
    """
    int course_nnpack_fft16(const float*,const float*,const float*,float*,int,int,int,int,int)
    void course_conv_forward(const float*,const float*,const float*,float*,int,int,int,int,int,int,int,int,int,int,int)
    void course_linear_split(const float*,const float*,const float*,float*,int,int,int)
    void course_bias_outer(const float*,float*,int,int)
    void course_conv_bias(const float*,float*,int,int,int,int)
    void course_conv_weight(const float*,const float*,float*,int,int,int,int,int)

def convolution(float[:,:,:,::1] inputs,float[:,:,:,::1] weights,float[::1] bias):
    if (inputs.shape[0] not in (2,16,32) or inputs.shape[1]!=1 or inputs.shape[2]!=28 or inputs.shape[3]!=28
        or weights.shape[0]!=8 or weights.shape[1]!=1 or weights.shape[2]!=5 or weights.shape[3]!=5 or bias.shape[0]!=8):
        raise ValueError('Native CNN convolution is restricted to verified small-preset shapes')
    result=np.empty((inputs.shape[0],8,24,24),dtype=np.float32)
    cdef float[:,:,:,::1] output=result
    cdef int status=0
    with nogil:
        if inputs.shape[0]>=16:
            status=course_nnpack_fft16(&inputs[0,0,0,0],&weights[0,0,0,0],&bias[0],&output[0,0,0,0],inputs.shape[0],1,28,28,8)
        else:
            course_conv_forward(&inputs[0,0,0,0],&weights[0,0,0,0],&bias[0],&output[0,0,0,0],inputs.shape[0],1,28,28,8,5,5,1,0,1,1)
    if status:raise MemoryError('CNN workspace could not allocate. Reduce the batch size and retry.')
    return result

def convolution_backward(float[:,:,:,::1] inputs,float[:,:,:,::1] gradient):
    if (inputs.shape[0] not in (2,16,32) or inputs.shape[1]!=1 or inputs.shape[2]!=28 or inputs.shape[3]!=28
        or gradient.shape[0]!=inputs.shape[0] or gradient.shape[1]!=8 or gradient.shape[2]!=24 or gradient.shape[3]!=24):
        raise ValueError('Native CNN backward dimensions differ from verified shapes')
    weights=np.empty((8,1,5,5),dtype=np.float32)
    bias=np.empty(8,dtype=np.float32)
    cdef float[:,:,:,::1] dw=weights
    cdef float[::1] db=bias
    with nogil:
        course_conv_weight(&inputs[0,0,0,0],&gradient[0,0,0,0],&dw[0,0,0,0],inputs.shape[0],1,28,28,8)
        course_conv_bias(&gradient[0,0,0,0],&db[0],gradient.shape[0],8,24,24)
    return weights,bias

def linear_forward(float[:,::1] inputs,float[:,::1] weights,float[::1] bias):
    if (inputs.shape[0] not in (2,16,32) or inputs.shape[1]!=weights.shape[1] or weights.shape[0]!=bias.shape[0]
        or (weights.shape[0],weights.shape[1]) not in ((64,288),(10,64))):
        raise ValueError('Native CNN linear dimensions differ from verified shapes')
    result=np.empty((inputs.shape[0],weights.shape[0]),dtype=np.float32)
    cdef float[:,::1] output=result
    with nogil:
        course_linear_split(&inputs[0,0],&weights[0,0],&bias[0],&output[0,0],inputs.shape[0],weights.shape[0],weights.shape[1])
    return result

def linear_bias(float[:,::1] gradient):
    if gradient.shape[1]!=64 or gradient.shape[0] not in (2,16,32):
        raise ValueError('Native CNN bias reduction dimensions differ from verified shapes')
    result=np.empty(64,dtype=np.float32)
    cdef float[::1] output=result
    with nogil:course_bias_outer(&gradient[0,0],&output[0],gradient.shape[0],64)
    return result
