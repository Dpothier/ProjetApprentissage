import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from networks.static_hypernetwork.modules.residual_block import ResidualBlock as ResidualBlockStatic
from networks.factorized_policy_hypernetwork_fast.modules.residual_block import ResidualBlock as ResidualBlockPolicy


torch.cuda.set_device(0)
batch_size = 256
input_size = 32
output_size = 32
w_x = 32
h_x = 32
w_k = 3
h_k = 3

static_residual_block = ResidualBlockStatic(in_size = input_size, out_size=output_size, downsample=False).cuda()
policy_residual_block = ResidualBlockPolicy(channels=input_size, downsample=False).cuda()

base_image = torch.randn((batch_size, input_size, w_x, h_x)).cuda()

static_weight1 = torch.randn((output_size, input_size, w_k, h_k)).cuda()
static_weight2 = torch.randn((output_size, input_size, w_k, h_k)).cuda()

policy_weight1 = torch.randn((batch_size, output_size, input_size, w_k, h_k)).cuda()
policy_weight2 = torch.randn((batch_size, output_size, input_size, w_k, h_k)).cuda()



out_static = static_residual_block(base_image, static_weight1, static_weight2)
out_policy = policy_residual_block(base_image, policy_weight1, policy_weight2)

