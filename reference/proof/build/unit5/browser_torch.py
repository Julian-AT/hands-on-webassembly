"""Pinned Borch CPU runtime shared by the original neural app staging copies."""
import borch as torch
import borchvision as torchvision
from torch_rng import install
from borch_compat import set_single_thread
from image_data import dataset_classes
from module_hooks import install as install_hooks
from image_transforms import install as install_transforms
install(torch)
install_hooks(torch)
install_transforms(torch, torchvision)
torch.set_num_threads=set_single_thread
torch.set_num_interop_threads=set_single_thread
if not hasattr(torch.nn.Module,'cpu'):
    torch.nn.Module.cpu=lambda self:self.to('cpu')
_deterministic=False
def use_deterministic_algorithms(mode,*,warn_only=False):
    global _deterministic
    if not isinstance(mode,bool) or not isinstance(warn_only,bool):
        raise TypeError('Determinism settings must be booleans')
    # This adapter uses only single-threaded CPU NumPy operations. There is no
    # alternative nondeterministic GPU implementation to select.
    _deterministic=mode
torch.use_deterministic_algorithms=use_deterministic_algorithms
torch.are_deterministic_algorithms_enabled=lambda:_deterministic
torchvision.datasets=dataset_classes(torch)
nn=torch.nn
T=torchvision.transforms
DataLoader=torch.utils.data.DataLoader
Subset=torch.utils.data.Subset
ConcatDataset=torch.utils.data.ConcatDataset

# Unit 5 calls torch.random directly and sets CUDA backend policy even though
# every model runs on CPU. Preserve that API surface without enabling a GPU.
from types import SimpleNamespace
if not hasattr(torch, 'random'):
    torch.random = SimpleNamespace()
torch.random.manual_seed = torch.manual_seed
if not hasattr(torch, 'backends'):
    torch.backends = SimpleNamespace()
if not hasattr(torch.backends, 'cudnn'):
    torch.backends.cudnn = SimpleNamespace(deterministic=False, benchmark=False, enabled=False)

# Borch's positional-only spelling differs from the original helper's calls.
# Retain the existing autograd implementations, including no-op squeeze axes.
if not getattr(torch, '_course_keywords_installed', False):
    _squeeze = torch.Tensor.squeeze
    def squeeze(self, dim=None):
        if dim is None:
            return _squeeze(self)
        return _squeeze(self, *dim) if isinstance(dim, (tuple, list)) else _squeeze(self, dim)
    torch.Tensor.squeeze = squeeze
    _cross_entropy_forward = nn.CrossEntropyLoss.forward
    def cross_entropy_forward(self, input, target):
        return _cross_entropy_forward(self, input, target)
    nn.CrossEntropyLoss.forward = cross_entropy_forward
    torch._course_keywords_installed = True

from neural_compat import install as install_neural_arithmetic
install_neural_arithmetic(torch)

from cnn_compat import install as install_cnn_arithmetic
install_cnn_arithmetic(torch)
