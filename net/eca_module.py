from torch import nn
import math

class ECABlock(nn.Module):
    """
    ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    https://doi.org/10.48550/arXiv.1910.03151
    https://github.com/BangguWu/ECANet
    """

    def __init__(self, n_channels, k_size=3, gamma=2, b=1):
        super(ECABlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
	
	# https://github.com/BangguWu/ECANet/issues/243 
	# dynamically computing the k_size 
        t = int(abs((math.log(n_channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
	
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        # feature descriptor on the global spatial information
        y = self.global_avg_pool(x)

        # y = self.conv(y.squeeze(-1).squeeze(-2).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-2).unsqueeze(-1) # b, c, z, h, w = x.size()
        y = self.conv(y.squeeze(-1).squeeze(-2).transpose(-2, -1)).transpose(-2, -1).unsqueeze(-2).unsqueeze(-1) # b, c, w, h, z = x.size()

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
    
if __name__ == "__main__":
    n_channels = 128
    b = 1
    gamma = 2
    t = int(abs((math.log(n_channels, 2) + b) / gamma))
    k_size = t if t % 2 else t + 1
    print(k_size)
    block = ECABlock(n_channels)
    import torchsummary
    print(block)
    torchsummary.summary(block, input=(4,128,16,16,16))