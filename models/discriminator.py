import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


def hinge_d_loss_with_exemplar_weights(logits_real, logits_fake, weights):
    assert weights.shape[0] == logits_real.shape[0] == logits_fake.shape[0]
    loss_real = torch.mean(F.relu(1.0 - logits_real), dim=[1, 2, 3])
    loss_fake = torch.mean(F.relu(1.0 + logits_fake), dim=[1, 2, 3])
    loss_real = (weights * loss_real).sum() / weights.sum()
    loss_fake = (weights * loss_fake).sum() / weights.sum()
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=4, patch_size=1):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        norm_layer = nn.BatchNorm2d
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        if patch_size == 2:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        elif patch_size == 1:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            raise NotImplementedError('patch size %d is not supported' % patch_size)
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=3, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)
        # self.main = nn.ModuleList(sequence)

        self.apply(weights_init)

    def forward(self, x_real, x_recon):
        """Standard forward."""
        x = torch.cat((x_real, x_recon), dim=1)
        # for m in self.main:
        #     x = m(x)
        #     print(x.shape)
        return self.main(x)
    

def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight

class LatentDiscriminator(nn.Module):
    def __init__(
            self, 
            disc_start,
            disc_in_channels, 
            patch_size, 
            disc_weight=0.5, 
            disc_factor=1.0, 
            disc_loss="hinge",
            ):
        super().__init__()
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels * 2, patch_size=patch_size)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    def forward(
        self,
        inputs,
        reconstructions,
        main_loss, 
        optimizer_idx,
        global_step,
        last_layer=None,
    ):
        # gen
        if optimizer_idx == 0:
            logits_fake = self.discriminator(reconstructions, inputs)
            g_loss = -torch.mean(logits_fake)
            if global_step >= self.discriminator_iter_start:
                if self.disc_factor > 0.0:
                    d_weight = self.calculate_adaptive_weight(
                        main_loss, g_loss, last_layer=last_layer
                    )
                else:
                    d_weight = torch.tensor(1.0)
            else:
                d_weight = torch.tensor(0.0)
                g_loss = torch.tensor(0.0, requires_grad=True)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            loss = d_weight * disc_factor * g_loss
            
            log = {
                "g_loss": g_loss.detach().mean().item(),
                "d_weight": d_weight.detach().item(),
                "gen_step_disc_factor": torch.tensor(disc_factor).item(),
            }
            return loss, log
        
        # discriminator training step
        elif optimizer_idx == 1:
            logits_real = self.discriminator(inputs.detach(), reconstructions.detach())
            logits_fake = self.discriminator(reconstructions.detach(), inputs.detach())

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )

            d_loss = self.disc_loss(logits_real, logits_fake)
            loss = disc_factor * d_loss

            log = {
                "disc_loss": d_loss.clone().detach().mean().item(),
                "disc_step_disc_factor": torch.tensor(disc_factor).item(),
                "logits_real": logits_real.detach().mean().item(),
                "logits_fake": logits_fake.detach().mean().item(),
            }
            return loss, log
    
if __name__ == '__main__':
    import torch
    disc_in_channels = 4
    disc_num_layers = 4
    disc_ndf = 64
    discriminator = NLayerDiscriminator(input_nc=disc_in_channels, n_layers=disc_num_layers, ndf=disc_ndf).cuda()
    x = torch.randn(16, disc_in_channels, 32, 32).cuda()
    print(discriminator)
    print(f"Discriminator Parameters: {sum(p.numel() for p in discriminator.parameters()) / 1e6:.2f}M")
    y = discriminator(x)
    print(y.shape)