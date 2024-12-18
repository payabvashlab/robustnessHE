import torch
import torch.nn as nn

from ..attack import Attack


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255):
        super().__init__("FGSM", model)
        self.eps = eps
        self.supported_mode = ["default", "targeted"]

    def dice_loss(self, pred, target, smooth=1e-6):
        # Flatten the tensors
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        # Calculate intersection and union
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        # Compute Dice coefficient
        dice = (2. * intersection + smooth) / (union + smooth)

        # Return Dice Loss
        return 1 - dice

    def forward(self, images, labels, loss):
        r"""
        Overridden.
        """
        minX = images.min()
        maxX = images.max()
        images = images.clone().detach().to(self.device)
        images01 = 1.0 * (images - images.min()) / (maxX - minX)
        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        images.requires_grad = True
        outputs = self.get_logits(images)
        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)
        grad = torch.autograd.grad(
                cost, images, retain_graph=False, create_graph=False
        )[0]
        adv_images = images01 + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        adv_images = minX + (maxX - minX) * (adv_images - adv_images.min()) / (adv_images.max() - adv_images.min())

        return adv_images
