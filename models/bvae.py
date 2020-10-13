import torch
from torch import nn, optim
from torch.nn import functional as F


class BetaVAE(nn.Module):

    num_iter = 0
    C_max = torch.Tensor([25])
    C_stop_iter = 10000
    gamma = 5.0
    beta = 5.0

    def __init__(self, input_size, encode_size, loss_type='B'):
        super(BetaVAE, self).__init__()
        self.input_size = input_size
        self.loss_type = loss_type
        self.fc1 = nn.Linear(input_size, 32)
        self.fc21 = nn.Linear(32, encode_size)
        self.fc22 = nn.Linear(32, encode_size)
        self.fc3 = nn.Linear(encode_size, 32)
        self.fc4 = nn.Linear(32, input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), x, mu, logvar

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        # Account for the minibatch samples from the dataset
        kld_weight = kwargs['M_N']

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2
                                               - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter *
                            self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'KLD': kld_loss}
