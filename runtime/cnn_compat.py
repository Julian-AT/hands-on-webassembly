"""Verified arithmetic for the original small CNN's tested image/batch shapes.

Only a marked convolution chain can select its linear kernels. Other models and
unverified shapes retain the existing adapter. This does not certify all CNNs.
"""
import sys


def install(torch):
    if sys.platform!='emscripten' or getattr(torch,'_course_cnn_installed',False):return
    import numpy as np
    from native_cnn import convolution,convolution_backward,linear_forward,linear_bias
    from native_linear import forward,bias_gradient,scaled_add
    original_make=torch.Tensor._make
    original_conv=torch.nn.Conv2d.forward
    original_linear=torch.nn.Linear.forward
    original_step=torch.optim.SGD.step
    shapes=((8,1,5,5),(8,),(64,288),(64,),(10,64),(10,))

    def pair(value):return (value,value) if isinstance(value,int) else tuple(value)
    def mark(module,enabled):
        module.weight._course_cnn_parameter=enabled
        if module.bias is not None:module.bias._course_cnn_parameter=enabled

    def make(self,data,parents,backward,op=None):
        result=original_make(self,data,parents,backward,op)
        if any(getattr(parent,'_course_small_cnn',False) for parent in parents):
            result._course_small_cnn=True
        return result

    def conv(self,x):
        supported=(self.bias is not None and x.data.dtype==np.float32 and self.weight.data.dtype==np.float32
            and x.shape in ((2,1,28,28),(16,1,28,28),(32,1,28,28)) and self.weight.shape==(8,1,5,5)
            and pair(self.stride)==(1,1) and pair(self.padding)==(0,0) and pair(self.dilation)==(1,1)
            and self.groups==1 and self.padding_mode=='zeros' and not x.requires_grad)
        mark(self,supported)
        if not supported:return original_conv(self,x)
        inputs=np.ascontiguousarray(x.data)
        values=convolution(inputs,np.ascontiguousarray(self.weight.data),np.ascontiguousarray(self.bias.data))
        def back(g):
            dw,db=convolution_backward(inputs,np.ascontiguousarray(g,dtype=np.float32))
            return np.zeros_like(inputs),dw,db
        result=x._make(values,(x,self.weight,self.bias),back,'NativeSmallConv')
        result._course_small_cnn=True
        return result

    def linear(self,x):
        supported=(getattr(x,'_course_small_cnn',False) and self.bias is not None
            and x.data.ndim==2 and x.data.dtype==np.float32 and self.weight.data.dtype==np.float32
            and x.shape[0] in (2,16,32) and self.weight.shape in ((64,288),(10,64)))
        mark(self,supported)
        if not supported:return original_linear(self,x)
        inputs=np.ascontiguousarray(x.data)
        values=linear_forward(inputs,np.ascontiguousarray(self.weight.data),np.ascontiguousarray(self.bias.data))
        def back(g):
            g=np.ascontiguousarray(g,dtype=np.float32)
            db=linear_bias(g) if g.shape[1]==64 else bias_gradient(g)
            return (forward(g,np.ascontiguousarray(self.weight.data.T),np.zeros(self.in_features,np.float32)),
                forward(np.ascontiguousarray(g.T),np.ascontiguousarray(inputs.T),np.zeros(self.in_features,np.float32)),db)
        return x._make(values,(x,self.weight,self.bias),back,'NativeLinearCandidate')

    def step(self):
        selected=[]
        for group in self.param_groups:
            params=group['params']
            if (tuple(p.shape for p in params)==shapes and all(getattr(p,'_course_cnn_parameter',False) for p in params)
                and not any(group[name] for name in ('weight_decay','dampening','nesterov','maximize'))):
                selected.append((group,[(p,p.data.copy()) for p in params if p.grad is not None]))
        original_step(self)
        for group,params in selected:
            for p,old in params:
                increment=self._state(p)['momentum_buffer'] if group['momentum'] else p.grad.data
                p._array=scaled_add(np.ascontiguousarray(old.reshape(-1)),np.ascontiguousarray(increment.reshape(-1)),np.float32(-group['lr'])).reshape(old.shape)

    torch.Tensor._make=make
    torch.nn.Conv2d.forward=conv
    torch.nn.Linear.forward=linear
    torch.optim.SGD.step=step
    torch._course_cnn_installed=True
