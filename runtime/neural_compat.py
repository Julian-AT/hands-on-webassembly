"""Verified arithmetic for the course's direct 784-input/10-class classifier.

Other model shapes retain the pinned adapter until separately certified.
"""
import sys


def install(torch):
    if sys.platform != 'emscripten' or getattr(torch,'_course_linear_installed',False):return
    import numpy as np
    from native_linear import forward,cross_entropy,bias_gradient,scaled_add
    original_forward=torch.nn.Linear.forward
    original_loss=torch.nn.CrossEntropyLoss.forward
    original_step=torch.optim.SGD.step

    def native_forward(self,x):
        if self.bias is None or x.data.ndim!=2 or x.data.dtype!=np.float32 or self.in_features!=784 or self.out_features!=10:
            return original_forward(self,x)
        result=forward(np.ascontiguousarray(x.data),np.ascontiguousarray(self.weight.data),np.ascontiguousarray(self.bias.data))
        def back(g):
            g=np.ascontiguousarray(g,dtype=np.float32)
            dx=forward(g,np.ascontiguousarray(self.weight.data.T),np.zeros(self.in_features,np.float32))
            dw=forward(np.ascontiguousarray(g.T),np.ascontiguousarray(x.data.T),np.zeros(self.in_features,np.float32))
            return dx,dw,bias_gradient(g)
        return x._make(result,(x,self.weight,self.bias),back,'NativeLinearCandidate')

    def native_loss(self,input,target):
        if input._op!='NativeLinearCandidate' or self.weight is not None or self.label_smoothing or self.reduction!='mean' or self.ignore_index!=-100:
            return original_loss(self,input,target)
        if np.any(target.data<0):return original_loss(self,input,target)
        loss,gradient=cross_entropy(np.ascontiguousarray(input.data),np.ascontiguousarray(target.data,dtype=np.int64))
        return input._make(np.asarray(loss),(input,),lambda g:(gradient*g,),'NativeCrossEntropyCandidate')

    def native_step(self):
        selected=[]
        for group in self.param_groups:
            params=group['params']
            if (len(params)==2 and params[0].shape==(10,784) and params[1].shape==(10,)
                and not any(group[name] for name in ('weight_decay','dampening','nesterov','maximize'))):
                selected.append((group,[(p,p.data.copy()) for p in params if p.grad is not None]))
        original_step(self)
        for group,saved in selected:
            for p,old in saved:
                increment=self._state(p)['momentum_buffer'] if group['momentum'] else p.grad.data
                p._array=scaled_add(np.ascontiguousarray(old.reshape(-1)),np.ascontiguousarray(increment.reshape(-1)),np.float32(-group['lr'])).reshape(old.shape)

    torch.nn.Linear.forward=native_forward
    torch.nn.CrossEntropyLoss.forward=native_loss
    torch.optim.SGD.step=native_step
    torch._course_linear_installed=True
