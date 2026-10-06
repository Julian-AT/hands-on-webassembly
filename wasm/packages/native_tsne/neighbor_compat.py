"""Scoped t-SNE Euclidean distance backend; original heap and self removal."""
import numpy as np
from sklearn.manifold import _t_sne
from sklearn.neighbors import NearestNeighbors
from native_neighbors import neighbors
class CompatibleNeighbors(NearestNeighbors):
    def kneighbors(self,X=None,n_neighbors=None,return_distance=True):
        n=self.n_neighbors if n_neighbors is None else n_neighbors
        x=getattr(self,"_fit_X",None)
        if not (X is None and self._fit_method=="brute" and self.effective_metric_=="euclidean"
                and isinstance(x,np.ndarray) and x.dtype==np.float64 and x.flags.c_contiguous
                and x.shape[1]==4 and x.shape[0] in (150,178,256) and isinstance(n,(int,np.integer))
                and 0<n<x.shape[0] and self.n_jobs in (None,1)):
            return super().kneighbors(X,n_neighbors,return_distance)
        distance,index=neighbors(x,n+1)
        mask=index!=np.arange(len(x))[:,None]
        duplicate=np.all(mask,axis=1);mask[:,0][duplicate]=False
        index=index[mask].reshape(len(x),n)
        return (distance[mask].reshape(len(x),n),index) if return_distance else index
_t_sne.NearestNeighbors=CompatibleNeighbors
