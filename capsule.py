'''
Created on Apr 3, 2018

@author: fox
'''

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def myCuda(input):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        return input.cuda()
    else:
        return input

class CapsNet(nn.Module):
    def __init__(self, A, B, C, D, E, iterations):
        super(CapsNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, A, 5, 2, 2)
        self.conv1_bn = nn.BatchNorm2d(A)
        
        self.primary_caps = PrimaryCaps(A, B)
        self.conv_capsule1 = ConvCaps(B, C, iterations, 2)
        self.conv_capsule2 = ConvCaps(C, D, iterations, 1)
        self.class_capsule = ClassCaps(D, E, iterations)
        
    def forward(self, x): # (bs, 1, 28, 28)
        x = F.relu(self.conv1_bn(self.conv1(x))) # (bs, 32, 14, 14)
        
        pose, activation = self.primary_caps(x) # (bs, 16, 4, 4, 14, 14) (bs, 16, 14, 14)
        pose, activation = self.conv_capsule1(pose, activation) # (bs, 16, 4, 4, 6, 6) (bs, 16, 6, 6)
        pose, activation = self.conv_capsule2(pose, activation) # (bs, 16, 4, 4, 4, 4) (bs, 16, 4, 4)
        pose, activation = self.class_capsule(pose, activation) # (bs, 10, 4, 4) (bs, 10)
        return activation
    
    def spread_loss(self, activation, target):
        '''
        activation: (bs, class_num)
        target: class_id LongTensor such as [1, 4, 9, ...]
        '''
        bs = activation.size(0)
        class_num = activation.size(1)
         
        y_gold = myCuda(autograd.Variable(torch.FloatTensor(bs, class_num))).zero_().scatter_(1, target.unsqueeze(1), 9999)
        loss = torch.clamp(0.2-(y_gold-activation), 0, 9999)
        loss = torch.sum(torch.pow(loss, 2), dim=1, keepdim=False).mean()

        return loss

    
class PrimaryCaps(nn.Module):
    def __init__(self, A, B):
        super(PrimaryCaps, self).__init__()
        
        self.pose_size = torch.Size([4,4])
        self.B = B
        
        self.conv_pose = nn.Conv2d(A, B*self.pose_size[0]*self.pose_size[1], 1, 1, 0)
        self.conv_activation = nn.Conv2d(A, B, 1, 1, 0)
        self.conv_activation_bn = nn.BatchNorm2d(B)
        
    def forward(self, x): # (bs, 32, 14, 14)
        bs = x.size(0)
        
        pose = self.conv_pose(x) # (bs, 16*4*4=256, 14, 14)
        pose = pose.view(bs, self.B, self.pose_size[0], self.pose_size[1], pose.size(-2), pose.size(-1))
        
        activation = F.sigmoid(self.conv_activation_bn(self.conv_activation(x))) # (bs, 16, 14, 14)

        return pose, activation
    
class ConvCaps(nn.Module):
    def __init__(self, in_cap_num, out_cap_num, iterations, stride):
        super(ConvCaps, self).__init__()
        self.in_cap_num = in_cap_num
        self.out_cap_num = out_cap_num
        self.iterations = iterations
        self.stride = stride
        self.kernel_size = 3 # 3x3
        self.pose_size = 4 # 4x4
        
        self.tile_filter_pose_inchannel = self.in_cap_num*self.pose_size*self.pose_size
        self.tile_filter_pose_outchannel = self.tile_filter_pose_inchannel*self.kernel_size*self.kernel_size
        self.tile_filter_pose = nn.Conv2d(self.tile_filter_pose_inchannel, self.tile_filter_pose_outchannel,
                                           self.kernel_size, self.stride, 0)
        
        self.tile_filter_activation_inchannel = self.in_cap_num
        self.tile_filter_activation_outchannel = self.tile_filter_activation_inchannel*self.kernel_size*self.kernel_size
        self.tile_filter_activation = nn.Conv2d(self.tile_filter_activation_inchannel, self.tile_filter_activation_outchannel,
                                                self.kernel_size, self.stride, 0)
        self.tile_filter_activation_bn = nn.BatchNorm2d(self.tile_filter_activation_outchannel)
        
        self.kernel_cap_num = self.in_cap_num*self.kernel_size*self.kernel_size
        # (16x9=144, 32, 4, 4)
        self.W = nn.Parameter(torch.randn(self.kernel_cap_num , self.out_cap_num, self.pose_size, self.pose_size))
        
        self.beta_u = nn.Parameter(torch.randn(self.out_cap_num))
        self.beta_a = nn.Parameter(torch.randn(self.out_cap_num))
        
        
        
    def forward(self, pose, activation): # (bs, 16, 4, 4, 14, 14) (bs, 16, 14, 14)
        bs = pose.size(0)
        # (bs, 16x4x4=256, 14, 14)
        pose = pose.contiguous().view(-1, self.tile_filter_pose_inchannel, pose.size(-2), pose.size(-1))
        pose = self.tile_filter_pose(pose) # (bs, 16x4x4x3x3=2304, 6, 6)
        # (bs, 144, 4, 4, 6, 6)
        pose = pose.view(bs, self.tile_filter_activation_outchannel, self.pose_size, self.pose_size, pose.size(-2), pose.size(-1))
        # (bs, 144, 6, 6)
        activation = F.sigmoid(self.tile_filter_activation_bn(self.tile_filter_activation(activation)))
        
        pose = pose.permute(0, 4, 5, 1, 2, 3) # (bs, 6, 6, 144, 4, 4)
        pose = pose.unsqueeze(dim=4) # (bs, 6, 6, 144, 1, 4, 4)
        W = self.W.view(1, 1, 1, self.kernel_cap_num , self.out_cap_num, self.pose_size, self.pose_size)
        vote = torch.matmul(pose, W) # (bs, 6, 6, 144, 16, 4, 4)
        vote = vote.view(bs, vote.size(1), vote.size(2), vote.size(3), vote.size(4), -1) # (bs, 6, 6, 144, 16, 16)
        
        activation = activation.permute(0, 2, 3, 1) # (bs, 6, 6, 144)
        
        # poses: (bs, 6, 6, 16, 16) , activations: (bs, 6, 6, 16)
        pose, activation = matrix_capsules_em_routing2(
              vote, activation, self.beta_u, self.beta_a, self.iterations)
        
        pose = pose.view(bs, pose.size(1), pose.size(2), self.out_cap_num, self.pose_size, self.pose_size)
        pose = pose.permute(0, 3, 4, 5, 1, 2)
        activation = activation.permute(0, 3, 1, 2)
        # poses: (bs, 16, 4, 4, 6, 6) , activations: (bs, 16, 6, 6)
        return pose, activation
    
class ClassCaps(nn.Module):
    def __init__(self, in_cap_num, out_cap_num, iterations):
        super(ClassCaps, self).__init__()
        
        self.in_cap_num = in_cap_num
        self.out_cap_num = out_cap_num
        self.iterations = iterations
        self.kernel_size = 1
        self.pose_size = 4 # 4x4
        
        self.kernel_cap_num = self.in_cap_num*self.kernel_size*self.kernel_size
        
        self.W = nn.Parameter(torch.randn(self.kernel_cap_num , self.out_cap_num, self.pose_size, self.pose_size))
        
        self.beta_u = nn.Parameter(torch.randn(self.out_cap_num))
        self.beta_a = nn.Parameter(torch.randn(self.out_cap_num))
        
    def forward(self, pose, activation): # (bs, 16, 4, 4, 4, 4) (bs, 16, 4, 4)
        
        bs = pose.size(0)
        
        pose = pose.permute(0, 4, 5, 1, 2, 3) # (bs, 4, 4, 16, 4, 4)
        pose = pose.unsqueeze(dim=4) # (bs, 4, 4, 16, 1, 4, 4)
        W = self.W.view(1, 1, 1, self.kernel_cap_num , self.out_cap_num, self.pose_size, self.pose_size)
        vote = torch.matmul(pose, W) # (bs, 4, 4, 16, 10, 4, 4)
        vote = vote.view(bs, -1, vote.size(4), self.pose_size*self.pose_size) # (bs, 256, 10, 16)
        
        activation = activation.contiguous().view(bs, -1) # (bs, 256)
        
        # poses: (bs, 10, 16) , activations: (bs, 10)
        pose, activation = matrix_capsules_em_routing2(
              vote, activation, self.beta_u, self.beta_a, self.iterations)
        
        pose = pose.view(bs, self.out_cap_num, self.pose_size, self.pose_size)
        # poses: (bs, 10, 4, 4) , activations: (bs, 10)
        return pose, activation
    


def matrix_capsules_em_routing1(V, a_, beta_v, beta_a, iterations): 
    if len(V.size())==6:
        bs = V.size(0)
        spatial = V.size(1)
        kernel_and_incap = V.size(3)
        outcap = V.size(4)
        R = myCuda(autograd.Variable(torch.ones(bs, spatial, spatial, kernel_and_incap, outcap), requires_grad=False)) / outcap
        beta_v = beta_v.view(1, 1, 1, 1, outcap, 1)
        beta_a = beta_a.view(1, 1, 1, outcap)
    else:
        bs = V.size(0)
        kernel_and_incap = V.size(1)
        outcap = V.size(2)
        R = myCuda(autograd.Variable(torch.ones(bs, kernel_and_incap, outcap), requires_grad=False)) / outcap
        beta_v = beta_v.view(1, 1, outcap, 1)
        beta_a = beta_a.view(1, outcap)

        
   
    epsilon = 1e-9
    it_min = 1.0
    it_max = min(iterations, 3.0)
    
    for it in range(iterations):
        inverse_temperature = it_min + (it_max - it_min) * it / max(1.0, iterations - 1.0)
        # M-step
        R = (R * a_.unsqueeze(-1)) # (bs, spatial, spatial, incap, outcap)
        sum_R = R.sum(-2, keepdim=True).unsqueeze(-1)  # (bs, spatial, spatial, 1, outcap, 1)
        mu = ((R.unsqueeze(-1) * V).sum(-3, keepdim=True) / (sum_R+epsilon)) # (bs, spatial, spatial, 1, outcap, capdim)
#         sigma_square = (R.unsqueeze(-1) * (V - mu) ** 2).sum(-3, keepdim=True) / (sum_R+epsilon) # (bs, spatial, spatial, 1, outcap, capdim)
        sigma_square = ((R.unsqueeze(-1) * (V - mu) ** 2).sum(-3, keepdim=True) / (sum_R+epsilon)) ** (1/2) # (bs, spatial, spatial, 1, outcap, capdim)
        # E-step
        if it != iterations - 1:
            mu, sigma_square, V_, a__ = mu.data, sigma_square.data, V.data, a_.data
            normal = Normal(mu, sigma_square)
            p = torch.exp(normal.log_prob(V_)) # (bs, spatial, spatial, incap, outcap, capdim)
            ap = a__.unsqueeze(-1) * p.sum(-1, keepdim=False) # (bs, spatial, spatial, incap, outcap)
            R = myCuda(autograd.Variable((ap / (ap.sum(-1, keepdim=True)+epsilon)).squeeze(), requires_grad=False))
        
    o_cost = ((beta_v+ torch.log(sigma_square+epsilon))* sum_R).sum(-1).squeeze() # (bs, spatial, spatial, outcap)
    # For numeric stability.
    o_cost_mean = torch.mean(o_cost, dim=-1, keepdim=True)
    o_cost_stdv = torch.sqrt(torch.abs(
        torch.sum((o_cost - o_cost_mean)*(o_cost - o_cost_mean), dim=-1, keepdim=True) 
            / (o_cost.size(-1)+epsilon)
        ))
    
    o_cost = beta_a + (o_cost_mean - o_cost) / (o_cost_stdv + epsilon)
    a = torch.sigmoid(inverse_temperature * (beta_a - o_cost))
    mu = mu.squeeze() 


    return mu, a  

def matrix_capsules_em_routing2(votes, i_activations, beta_v, beta_a, iterations): 
    '''
    For ConvCaps, the dimensions of parameters are former.
    For ClassCaps, the dimensions of parameters are latter.
    votes: (bs, spatial, spatial, incap, outcap, cap_dim) or (bs, incap, outcap, capdim)
    i_activations: (bs, spatial, spatial, incap) or  (bs, incap)
    beta_v, beta_a: (outcap)
    
    return:
    poses: (bs, spatial, spatial, outcap, cap_dim) or (bs, outcap, cap_dim)
    activations: (bs, spatial, spatial, outcap) or (bs, outcap)
    '''
    
    if len(votes.size())==6:
        bs = votes.size(0)
        spatial = votes.size(1)
        kernel_and_incap = votes.size(3)
        outcap = votes.size(4)
        # Rij in the paper, (bs, spatial, spatial, incap, outcap)
        rr = myCuda(autograd.Variable(torch.ones(bs, spatial, spatial, kernel_and_incap, outcap), requires_grad=False)) / outcap
        beta_v = beta_v.view(1, 1, 1, 1, outcap, 1)
        beta_a = beta_a.view(1, 1, 1, outcap)
    else:
        bs = votes.size(0)
        kernel_and_incap = votes.size(1)
        outcap = votes.size(2)
        rr = myCuda(autograd.Variable(torch.ones(bs, kernel_and_incap, outcap), requires_grad=False)) / outcap
        beta_v = beta_v.view(1, 1, outcap, 1)
        beta_a = beta_a.view(1, outcap)
    
    it_min = 1.0
    it_max = min(iterations, 3.0)
    epsilon = 1e-9
    
    for it in range(iterations):
        
        inverse_temperature = it_min + (it_max - it_min) * it / max(1.0, iterations - 1.0)
        # m-step
        rr_prime = (rr * i_activations.unsqueeze(-1)) # (bs, spatial, spatial, incap, outcap)
        rr_prime_sum = rr_prime.sum(-2, keepdim=True).unsqueeze(-1)  # (bs, spatial, spatial, 1, outcap, 1)
        o_mean = ((rr_prime.unsqueeze(-1) * votes).sum(-3, keepdim=True) / (rr_prime_sum+epsilon)) # (bs, spatial, spatial, 1, outcap, capdim)
        o_stdv = ((rr_prime.unsqueeze(-1) * (votes - o_mean) ** 2).sum(-3, keepdim=True) / (rr_prime_sum+epsilon)) ** (1/2) # (bs, spatial, spatial, 1, outcap, capdim)
        o_cost = ((beta_v+ torch.log(o_stdv+epsilon))* rr_prime_sum).sum(-1).squeeze() # (bs, spatial, spatial, outcap)
        # For numeric stability.
        o_cost_mean = torch.mean(o_cost, dim=-1, keepdim=True)
        o_cost_stdv = torch.sqrt(torch.sum((o_cost - o_cost_mean)**2, dim=-1, keepdim=True) / outcap)
        # (bs, spatial, spatial, outcap)
        o_activations_cost = beta_a + (o_cost_mean - o_cost) / (o_cost_stdv + epsilon)
        o_activations = torch.sigmoid(inverse_temperature * o_activations_cost)
   
        # e-step
        if it < iterations - 1:
            mu, sigma_square, V_, a__ = o_mean.data, o_stdv.data, votes.data, o_activations.data
            normal = Normal(mu, sigma_square)
            p = torch.exp(normal.log_prob(V_)) # (bs, spatial, spatial, incap, outcap, capdim)
            ap = a__.unsqueeze(-2) * p.sum(-1, keepdim=False) # (bs, spatial, spatial, incap, outcap)
            rr = myCuda(autograd.Variable(ap / (ap.sum(-1, keepdim=True)+epsilon), requires_grad=False))
            

    return o_mean.squeeze(), o_activations



def matrix_capsules_em_routing(votes, i_activations, beta_v, beta_a, iterations):
    '''
    For ConvCaps, the dimensions of parameters are former.
    For ClassCaps, the dimensions of parameters are latter.
    votes: (bs, spatial, spatial, incap, outcap, cap_dim) or (bs, incap, outcap, capdim)
    i_activations: (bs, spatial, spatial, incap) or  (bs, incap)
    beta_v, beta_a: (outcap)
    
    return:
    poses: (bs, spatial, spatial, outcap, cap_dim) or (bs, outcap, cap_dim)
    activations: (bs, spatial, spatial, outcap) or (bs, outcap)
    '''
    
    if len(votes.size())==6:
        bs = votes.size(0)
        spatial = votes.size(1)
        kernel_and_incap = votes.size(3)
        outcap = votes.size(4)
        pose_size = votes.size(5)
        b_convcaps = True
    else:
        bs = votes.size(0)
        kernel_and_incap = votes.size(1)
        outcap = votes.size(2)
        pose_size = votes.size(3)
        b_convcaps = False
        
    # Rij in the paper, (bs, 6, 6, 288, 32, 16)
    rr = myCuda(autograd.Variable(torch.ones(votes.size()), requires_grad=False))
    rr *= 1/outcap
    
    # (bs, 6, 6, 288, 32, 16)
    if b_convcaps:
        i_activations = i_activations.contiguous().view(bs, spatial, spatial, kernel_and_incap, 1, 1)
        i_activations = i_activations.expand(-1, -1, -1, -1, outcap, pose_size)
    else:
        i_activations = i_activations.contiguous().view(bs, kernel_and_incap, 1, 1)
        i_activations = i_activations.expand(-1, -1, outcap, pose_size)
    
    if b_convcaps:
        # (bs, 6, 6, 1, 32, 16)
        beta_v = beta_v.view(1, 1, 1, 1, outcap, 1).expand(bs, spatial, spatial, -1, -1, pose_size)
        # (bs, 6, 6, 1, 32, 1)
        beta_a = beta_a.view(1, 1, 1, 1, outcap, 1).expand(bs, spatial, spatial, -1, -1, -1)
    else:
        beta_v = beta_v.view(1, 1, outcap, 1).expand(bs, -1, -1, pose_size)
        beta_a = beta_a.view(1, 1, outcap, 1).expand(bs, -1, -1, -1)
            
    # inverse_temperature schedule (min, max)
    it_min = 1.0
    it_max = min(iterations, 3.0)
    for it in range(iterations):
        
        inverse_temperature = it_min + (it_max - it_min) * it / max(1.0, iterations - 1.0)
        o_mean, o_stdv, o_activations = m_step(
            rr, votes, i_activations, beta_v, beta_a, inverse_temperature, b_convcaps
            )
        if it < iterations - 1:
            rr.data = e_step(
                o_mean, o_stdv, o_activations, votes, b_convcaps
                ).data


    # (bs, 6, 6, 32, 16)
    poses = o_mean.squeeze(dim=-3)
    # [bs, 6, 6, 32]
    activations = o_activations.squeeze()

    return poses, activations
    
    
def m_step(rr, votes, i_activations, beta_v, beta_a, inverse_temperature, b_convcaps):
    epsilon = 1e-9
    # (bs, 6, 6, 288, 32, 16)
    rr_prime = rr * i_activations
    # (bs, 6, 6, 1, 32, 16)
    rr_prime_sum = torch.sum(rr_prime, dim=-3, keepdim=True)
    # (bs, 6, 6, 1, 32, 16)
    o_mean = torch.sum(rr_prime * votes, dim=-3, keepdim=True) / (rr_prime_sum+epsilon)
    if b_convcaps:
        votes_sub_mean = votes - o_mean.expand(-1,-1,-1,votes.size(-3), -1, -1)
    else:
        votes_sub_mean = votes - o_mean.expand(-1,votes.size(-3), -1, -1)
    # (bs, 6, 6, 1, 32, 16)
    o_stdv = torch.sqrt(torch.abs(
        torch.sum(rr_prime * votes_sub_mean * votes_sub_mean, dim=-3, keepdim=True) / (rr_prime_sum+epsilon)
        ))
    # (bs, 6, 6, 1, 32, 16)
    o_cost_h = (beta_v + torch.log(o_stdv + epsilon)) * rr_prime_sum
    
    # o_cost: (bs, 6, 6, 1, 32, 1)
    o_cost = torch.sum(o_cost_h, dim=-1, keepdim=True)
    # For numeric stability.
    if b_convcaps:
        o_cost_mean = torch.mean(o_cost, dim=-2, keepdim=True).expand(-1, -1, -1, -1, o_cost.size(-2), -1)
        o_cost_stdv = torch.sqrt(torch.abs(
            torch.sum((o_cost - o_cost_mean)*(o_cost - o_cost_mean), dim=-2, keepdim=True) 
                / (o_cost.size(-2)+epsilon)
            )).expand(-1, -1, -1, -1, o_cost.size(-2), -1)
    else:
        o_cost_mean = torch.mean(o_cost, dim=-2, keepdim=True).expand(-1, -1, o_cost.size(-2), -1)
        o_cost_stdv = torch.sqrt(torch.abs(
            torch.sum((o_cost - o_cost_mean)*(o_cost - o_cost_mean), dim=-2, keepdim=True) 
                / (o_cost.size(-2)+epsilon)
            )).expand(-1, -1, o_cost.size(-2), -1)        
    # (bs, 6, 6, 1, 32, 1)
    o_activations_cost = beta_a + (o_cost_mean - o_cost) / (o_cost_stdv + epsilon)
    
    # (bs, 6, 6, 1, 32, 1)
    o_activations = torch.sigmoid(inverse_temperature * o_activations_cost)
    
    return o_mean, o_stdv, o_activations

def e_step(o_mean, o_stdv, o_activations, votes, b_convcaps):
    epsilon = 1e-9
    
    if b_convcaps:
        # (bs, 6, 6, 288, 32, 16)
        o_mean = o_mean.expand(-1,-1,-1,votes.size(-3), -1, -1)
        o_stdv = o_stdv.expand(-1,-1,-1,votes.size(-3), -1, -1)
    else:
        o_mean = o_mean.expand(-1,votes.size(-3), -1, -1)
        o_stdv = o_stdv.expand(-1,votes.size(-3), -1, -1)
    # (bs, 6, 6, 288, 32, 1)
    o_p_unit0 = -torch.sum((votes-o_mean)*(votes-o_mean) / (2 * o_stdv*o_stdv+epsilon), dim=-1, keepdim=True)
    o_p_unit2 = -torch.sum(torch.log(o_stdv + epsilon), dim=-1, keepdim=True)
    
    # p_j in the paper, (bs, 6, 6, 288, 32, 1)
    o_p = o_p_unit0 + o_p_unit2
    
    # (bs, 6, 6, 288, 32, 1)
    if b_convcaps:
        zz = torch.log(o_activations.expand(-1, -1, -1, votes.size(-3), -1, -1) + epsilon) + o_p
#         rr = F.softmax(zz, dim=-2)
        rr = F.log_softmax(zz, dim=-2)
        rr = rr.expand(-1, -1, -1, -1, -1, votes.size(-1))
    else:
        zz = torch.log(o_activations.expand(-1, votes.size(-3), -1, -1) + epsilon) + o_p
#         rr = F.softmax(zz, dim=-2)
        rr = F.log_softmax(zz, dim=-2)
        rr = rr.expand(-1, -1, -1, votes.size(-1))        
    
    return rr
        
