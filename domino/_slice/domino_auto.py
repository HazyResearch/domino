from torch import nn
from torch.nn import functional as F
import torch
from tqdm  import tqdm


class DominoAutoEncoder(nn.Module):
    def __init__(self, n_slices: int, embedding_dim: int, target_dim: int, alpha: float):
        super().__init__()
        self.n_slices
        self.embedding_dim = embedding_dim
        self.target_dim = target_dim
        self.alpha = alpha
        self.encoder = nn.Linear(embedding_dim, n_slices)
        self.decoder = nn.Linear(embedding_dim, embedding_dim + target_dim)

    def forward(self, x: torch.Tensor):
        return self.encoder(x)

    def fit(self, x: torch.Tensor, l: torch.Tensor):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.alpha)

        with tqdm(total=self.n_epochs) as pbar:
            for epoch in range(self.n_epochs):
                batcher = lambda data: torch.split(data, self.batch_size, dim=0)
                for x_batch, l_batch in zip(batcher(x), batcher(l)):

                    s_batch = self.encoder(x_batch)
                    out = self.decoder(s_batch)
                    x_hat = out[:, : self.embedding_dim]
                    l_hat = out[:, self.target_dim :]

                    loss = (
                        self.alpha * F.binary_cross_entropy_with_logits(l_hat, l_batch) +
                        (1 - self.alpha) F.mse_loss(x_hat, x_batch) + 
                    )

                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()
                    
                pbar.update()
                pbar.set_postfix(epoch=epoch, loss=loss.item())

    def predict_proba(self, x: torch.Tensor):
        return self.encoder(x)
    
    def predict(self, x: torch.Tensor):
        return self.encoder(x) > 0.5