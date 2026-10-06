"""The supplied CNN app flips CHW tensors after ToTensor, using torch's RNG."""


def install(torch, torchvision):
    from PIL import Image

    def flip_call(dimension, pil_operation):
        def call(self, image):
            # Consume a draw even for p=0 or p=1, as torchvision does.
            if not (torch.rand(1) < self.p).item():
                return image
            if isinstance(image, torch.Tensor):
                return torch.flip(image, dims=[dimension])
            if isinstance(image, Image.Image):
                return image.transpose(pil_operation)
            raise TypeError('Expected a Tensor or PIL image')
        return call

    torchvision.transforms.RandomHorizontalFlip.__call__ = flip_call(-1, Image.Transpose.FLIP_LEFT_RIGHT)
    torchvision.transforms.RandomVerticalFlip.__call__ = flip_call(-2, Image.Transpose.FLIP_TOP_BOTTOM)
