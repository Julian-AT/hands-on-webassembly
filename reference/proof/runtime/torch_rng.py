"""CPU RNG compatibility for the float32 course models and zero-worker loaders.

Reference: frozen PyTorch CPU MT19937 and ATen TransformationHelper /
DistributionsHelper. NumPy RandomState supplies the same raw MT19937 words;
NumPy's distribution/sampling functions must NOT supply the transformed draws.
Normal distributions and unsupported sampling options fail explicitly.
"""
import numpy as np


class CPUStream:
    def __init__(self, seed=0): self.manual_seed(seed)
    def manual_seed(self, seed):
        seed = int(seed)
        if not -(2**63) <= seed < 2**64: raise RuntimeError('Seed out of range')
        self.seed = seed % 2**64
        self.state = np.random.RandomState(self.seed % 2**32)
        return self
    def words(self, size=None): return self.state.randint(0,2**32,size=size,dtype=np.uint32)
    def words64(self, size=None):
        shape = () if size is None else (size,) if isinstance(size,int) else tuple(size)
        words = self.words(shape+(2,)).astype(np.uint64)
        return (words[...,0] << np.uint64(32)) | words[...,1]
    def random(self, size=None):
        return (self.words(size) & np.uint32(2**24-1)).astype(np.float32)*np.float32(2**-24)
    def uniform(self, low=0, high=1, size=None):
        low,high=np.float32(low),np.float32(high)
        x=self.random(size)
        # ARM's native compiler may fuse multiply-add. Float64 calculation here
        # emulates a single rounding of the float32 operands.
        result=(x.astype(np.float64)*float(np.float32(high-low))+float(low)).astype(np.float32)
        return np.where(result==high,low,result)
    def bernoulli(self, p, size):
        x=(self.words64(size)&np.uint64(2**53-1)).astype(np.float64)*2**-53
        return x < p
    def seed64(self): return int(self.words64() & np.uint64(2**63-1))
    def permutation(self,n):
        if not isinstance(n,(int,np.integer)) or not 0 <= n < 2**32//20:
            raise ValueError('Only small integer randperm is supported by the course adapter')
        result=np.arange(n,dtype=np.int64)
        # PyTorch draws n-1 words, and performs forward Fisher-Yates swaps.
        raw=self.words(max(0,n-1))
        for i,word in enumerate(raw):
            j=i+int(word)%(n-i)
            result[i],result[j]=result[j],result[i]
        return result
    def standard_normal(self,*args,**kwargs):
        raise NotImplementedError('Exact CPU normal sampling has not been implemented')


def install(torch):
    """Patch the pinned Borch package once, retaining its autograd and layers."""
    if getattr(torch,'_course_rng_installed',False): return
    import borch._ops as ops
    import borch._nn as nn
    import borch._data as data
    from borch_compat import set_single_thread
    torch.set_num_threads = set_single_thread
    torch.set_num_interop_threads = set_single_thread
    stream=CPUStream()
    for module in (torch,ops,nn,data): module._rng=stream
    class Generator:
        def __init__(self): self._rng=CPUStream()
        def manual_seed(self,seed): self._rng.manual_seed(seed);return self
        def initial_seed(self): return self._rng.seed
        def rng(self): return self._rng
    def manual_seed(seed):
        stream.manual_seed(seed)
        ops._LAST_SEED[0]=stream.seed
        return default_generator
    default_generator=Generator()
    default_generator._rng=stream
    torch.Generator=ops.Generator=Generator
    torch.manual_seed=ops.manual_seed=manual_seed
    torch.default_generator=default_generator
    if hasattr(torch,'random'): torch.random.manual_seed=manual_seed
    def dropout_body(input,p,training):
        if not 0<=p<=1: raise ValueError('dropout probability must lie in [0,1]')
        if not training or p==0 or input.numel()==0: return input
        if p==1: return input*0
        mask=stream.bernoulli(1-p,tuple(input.shape)).astype(np.float32)/np.float32(1-p)
        return input*torch.tensor(mask)
    ops._dropout_body=dropout_body
    class RandomSampler(data.RandomSampler):
        def __iter__(self):
            if self.replacement: raise NotImplementedError('Replacement sampling is outside the course adapter')
            rng=self.generator.rng() if self.generator is not None else CPUStream(stream.seed64())
            n=len(self.data_source)
            for _ in range(len(self)//n): yield from rng.permutation(n).tolist()
            # PyTorch consumes the final permutation even when its slice is empty.
            yield from rng.permutation(n).tolist()[:len(self)%n]
    data.RandomSampler=RandomSampler
    torch.utils.data.RandomSampler=RandomSampler
    original_loader=data.DataLoader
    class DataLoader(original_loader):
        def __init__(self,*args,num_workers=0,worker_init_fn=None,**kwargs):
            if len(args)>5: raise TypeError('Worker arguments must be keywords')
            if num_workers!=0: raise ValueError('Only num_workers=0 is supported')
            super().__init__(*args,num_workers=0,worker_init_fn=None,**kwargs)
        def __iter__(self):
            # _BaseDataLoaderIter consumes base_seed at iter(loader), before the
            # lazy sampler generates its indices at next(iterator).
            rng=self.generator.rng() if self.generator is not None else stream
            rng.seed64()
            return super().__iter__()
    torch.utils.data.DataLoader=data.DataLoader=DataLoader
    torch._course_rng_installed=True
    return stream
