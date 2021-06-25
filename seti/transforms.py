import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch


class RandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, data):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        img = data['img']
        target = data['target'] 
        if target == 1:
            if torch.rand(1) < self.p:
                
                return {'img' : F.hflip(img), 'target':target}
       
        
        return {'img':img, 'target': target}


    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, data):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        img = data['img']
        target = data['target'] 
        if target == 1:
            if torch.rand(1) < self.p:
                
                return {'img' : F.vflip(img), 'target':target}
       
        
        return img


    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)







# test_transform = transforms.Compose([
# 	RandomHorizontalFlip(0.5),
#     RandomVerticallFlip(0.5)
# 	])


# x = torch.Tensor(3,32,32)
# x = test_transform({'img':x,'target':1})
# print(x['target'])