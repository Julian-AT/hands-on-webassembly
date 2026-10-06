import numpy as np
cimport numpy as cnp
from libc.math cimport fma,sqrt
from libc.float cimport DBL_MAX
ctypedef Py_ssize_t intp_t
include "_heap.pyx.pxi"
include "_sorting.pyx.pxi"

def operations(double[:,::1] x):
    if x.shape[1]!=4: raise ValueError("Only independently verified four-feature arithmetic")
    cdef intp_t n=x.shape[0],i,j,k
    cdef double[:,::1] mid=np.empty((n,n),dtype=np.float64)
    cdef double[::1] norm=np.empty(n,dtype=np.float64)
    cdef double a,b,s
    cdef intp_t tail8=(n//8)*8,tail4=(n//4)*4
    for i in range(n):
        a=x[i,0]*x[i,0];b=x[i,1]*x[i,1]
        a=a+x[i,2]*x[i,2];b=b+x[i,3]*x[i,3]
        norm[i]=a+b
        for j in range(n):
            s=0.
            for k in range(4):s=fma(x[i,k],x[j,k],s)
            if i>=tail8 and j>=tail8 and (i>=tail4 or j>=tail4):
                a=x[i,0]*x[j,0];b=x[i,1]*x[j,1]
                a=a+x[i,2]*x[j,2];b=b+x[i,3]*x[j,3];s=a+b
            mid[i,j]=-2.*s
    return np.asarray(norm),np.asarray(mid)

def neighbors(double[:,::1] x,intp_t count):
    if x.shape[0] not in (150,178,256) or not 1<=count<=x.shape[0]: raise ValueError("Verified single-chunk heap bounds")
    cdef intp_t n=x.shape[0],i,j
    norms,middle=operations(x)
    cdef double[::1] norm=norms
    cdef double[:,::1] mid=middle
    cdef double[:,::1] distances=np.full((n,count),DBL_MAX,dtype=np.float64)
    cdef intp_t[:,::1] indices=np.full((n,count),-1,dtype=np.intp)
    cdef double[:,::1] local_distances=np.full((n,count),DBL_MAX,dtype=np.float64)
    cdef intp_t[:,::1] local_indices=np.full((n,count),-1,dtype=np.intp)
    cdef double distance
    for i in range(n):
        for j in range(n):
            distance=(norm[i]+mid[i,j])+norm[j]
            heap_push(&local_distances[i,0],&local_indices[i,0],count,max(0.,distance),j)
        # Native auto strategy merges the one private Y heap into the main heap.
        for j in range(count):
            heap_push(&distances[i,0],&indices[i,0],count,local_distances[i,j],local_indices[i,j])
        simultaneous_sort(&distances[i,0],&indices[i,0],count)
        for j in range(count):distances[i,j]=sqrt(max(0.,distances[i,j]))
    return np.asarray(distances),np.asarray(indices)
