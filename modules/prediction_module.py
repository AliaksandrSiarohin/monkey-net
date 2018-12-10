from torch import nn
import torch


class PredictionModule(nn.Module):
    """
    RNN for predicting kp movement
    """

    def __init__(self, num_kp=10, kp_variance=0.01, num_features=1024, num_layers=1, dropout=0.5):
        super(PredictionModule, self).__init__()

        input_size = num_kp * (2 + 4 * (kp_variance == 'matrix'))

        self.rnn = nn.GRU(input_size=input_size, hidden_size=num_features, num_layers=num_layers,
                          dropout=dropout, batch_first=True)
        self.linear = nn.Linear(num_features, input_size)

    def net(self, input, h=None):
        output, h = self.rnn(input, h)
        init_shape = output.shape
        output = output.contiguous().view(-1, output.shape[-1])
        output = self.linear(output)
        return output.view(init_shape[0], init_shape[1], output.shape[-1]), h

    def forward(self, kp_batch):
        bs, d, num_kp, _ = kp_batch['mean'].shape
        inputs = [kp_batch['mean'].contiguous().view(bs, d, -1)]
        if 'var' in kp_batch:
            inputs.append(kp_batch['var'].contiguous().view(bs, d, -1))

        input = torch.cat(inputs, dim=-1)

        output, h = self.net(input)
        output = output.view(bs, d, num_kp, -1)
        mean = torch.tanh(output[:, :, :, :2])
        kp_array = {'mean': mean}
        if 'var' in kp_batch:
            var = output[:, :, :, 2:]
            var = var.view(bs, d, num_kp, 2, 2)
            var = torch.matmul(var.permute(0, 1, 2, 4, 3), var)
            kp_array['var'] = var

        return kp_array
