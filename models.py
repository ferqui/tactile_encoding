# from turtle import forward
import torch
import torch.nn as nn
import numpy as np

from collections import namedtuple

class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements 
    the surrogate gradient. By subclassing torch.autograd.Function, 
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid 
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 100

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad

activation = SurrGradSpike.apply


## Encoder
class Encoder(nn.Module):
    def __init__(self, nb_inputs, encoder_weight_scale, nb_input_copies):
        super(Encoder, self).__init__()

        enc_gain = torch.empty((nb_inputs,), requires_grad=False)
        enc_bias = torch.empty((nb_inputs,), requires_grad=False)
        torch.nn.init.normal_(enc_gain, mean=0.0, std=encoder_weight_scale)  # TODO update this parameter
        torch.nn.init.normal_(enc_bias, mean=0.0, std=1.0)

        self.nb_input_copies = nb_input_copies
        self.register_buffer('enc_gain', enc_gain)
        self.register_buffer('enc_bias', enc_bias)

    def forward(self, inputs):
        encoder_currents = self.enc_gain * (inputs.tile((self.nb_input_copies,)) + self.enc_bias) 
        return encoder_currents


## Mihilas-Niebur neuron
class MN_neuron(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'i1', 'i2', 'Thr', 'spk'])

    def __init__(self, nb_inputs, parameters_combination, dt=1/1000, a=5, A1=10, A2=-0.6, b=10, G=50, k1=200, k2=20, R1=0, R2=1, train=True): # default combination: M2O of the original paper
        super(MN_neuron, self).__init__()

        # One-to-one synapse
        self.linear = nn.Parameter(torch.ones(1, nb_inputs), requires_grad=train)

        self.C = 1

        self.N = nb_inputs

        self.EL = -0.07
        self.Vr = -0.07
        self.Tr = -0.06
        self.Tinf = -0.05

        self.a = a
        self.A1 = A1
        self.A2 = A2
        self.b = b  # units of 1/s
        self.G = G * self.C  # units of 1/s
        self.k1 = k1  # units of 1/s
        self.k2 = k2  # units of 1/s
        self.R1 = R1
        self.R2 = R2

        self.dt = dt # get dt from sample rate!

        parameters_list = ["a", "A1", "A2", "b", "G", "k1", "k2", "R1", "R2"]
        for ii in parameters_list:
            if ii in list(parameters_combination.keys()):
                eval_string = "self.{}".format(ii) + " = " + str(parameters_combination[ii])
                exec(eval_string)

        one2N_matrix = torch.ones(1, nb_inputs)

        self.a = nn.Parameter(one2N_matrix * self.a, requires_grad=train)
        
        self.A1 = nn.Parameter(one2N_matrix * self.A1 * self.C, requires_grad=train)
        self.A2 = nn.Parameter(one2N_matrix * self.A2 * self.C, requires_grad=train)

        self.state = None

    def forward(self, x):
        if self.state is None:
            self.state = self.NeuronState(V=torch.ones(x.shape[0], self.N, device=x.device) * self.EL,
                                          i1=torch.zeros(x.shape[0], self.N, device=x.device),
                                          i2=torch.zeros(x.shape[0], self.N, device=x.device),
                                          Thr=torch.ones(x.shape[0], self.N, device=x.device) * self.Tinf,
                                          spk=torch.zeros(x.shape[0], self.N, device=x.device))

        V = self.state.V
        i1 = self.state.i1
        i2 = self.state.i2
        Thr = self.state.Thr

        i1 += -self.k1 * i1 * self.dt
        i2 += -self.k2 * i2 * self.dt
        V += self.dt * (self.linear * x + i1 + i2 - self.G * (V - self.EL)) / self.C
        Thr += self.dt * (self.a * (V - self.EL) - self.b * (Thr - self.Tinf))

        spk = activation(V - Thr)

        i1 = (1 - spk) * i1 + (spk) * (self.R1 * i1 + self.A1)
        i2 = (1 - spk) * i2 + (spk) * (self.R2 * i2 + self.A2)
        Thr = (1 - spk) * Thr + (spk) * torch.max(Thr, torch.tensor(self.Tr))
        V = (1 - spk) * V + (spk) * self.Vr

        self.state = self.NeuronState(V=V, i1=i1, i2=i2, Thr=Thr, spk=spk)

        return spk

    def reset(self):
        self.state = None

# Izhikevich neuron
class IZ_neuron(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'i', 'spk'])
    def __init__(self, nb_inputs, parameters_combination, dt=1/1000, a=0.02, b=0.2, d=8, tau=1, k=150, train=True): # default combination: M2O of the original paper
        super(MN_neuron, self).__init__()

        # One-to-one synapse
        self.linear = nn.Parameter(torch.ones(1, nb_inputs), requires_grad=train)

        self.N = nb_inputs

        self.Vr = -0.07
        self.Tr = -0.03

        self.a = a
        self.b = b  # units of 1/s
        self.d = self.d
        self.k = k
        self.tau = tau

        self.dt = dt # get dt from sample rate!

        parameters_list = ["a", "b", "d", "tau", "k"]
        for ii in parameters_list:
            if ii in list(parameters_combination.keys()):
                eval_string = "self.{}".format(ii) + " = " + str(parameters_combination[ii])
                exec(eval_string)

        self.state = None
    
    def forward(self, x):
        if self.state is None:
            self.state = self.NeuronState(V=torch.ones(x.shape[0], self.N, device=x.device) * self.Vr,
                                          i=torch.zeros(x.shape[0], self.N, device=x.device),
                                          spk=torch.zeros(x.shape[0], self.N, device=x.device))
        
        V = self.state.V
        i = self.state.i

        V = self.dt * (V + self.tau * (0.04 * (V**2) + (5 * V) + 140 - i + x))
        i = self.dt * (i + self.tau * (self.a * ((self.b * V) - i)))

        spk = activation(V - self.Tr)

        i = (1 - spk) * i + (spk) * (i + self.d)
        V = (1 - spk) * V + (spk) * self.Vr

        self.state = self.NeuronState(V=V, i=i, spk=spk)

        return spk

    def reset(self):
        self.state = None


## LIF neuron
class LIF_neuron(nn.Module):
    LIFstate = namedtuple('LIFstate', ['syn', 'mem', 'S'])

    def __init__(self, nb_inputs, nb_outputs, alpha, beta, is_recurrent=True, fwd_weight_scale=1.0, rec_weight_scale=1.0):
        super(LIF_neuron, self).__init__()

        #weight = torch.empty((nb_inputs, nb_outputs))
        self.weight = torch.nn.Parameter(torch.empty((nb_inputs, nb_outputs)), requires_grad=True)
        torch.nn.init.normal_(self.weight, mean=0.0, std=fwd_weight_scale/np.sqrt(nb_inputs))

        self.is_recurrent = is_recurrent
        if is_recurrent:
            #weight_rec = torch.empty((nb_outputs, nb_outputs))
            #torch.nn.init.normal_(weight_rec, mean=0.0, std=rec_weight_scale / np.sqrt(nb_outputs))
            self.weight_rec = torch.nn.Parameter(torch.empty((nb_outputs, nb_outputs)), requires_grad=True)
            #torch.nn.init.zeros_(self.weight_rec)
            torch.nn.init.normal_(self.weight_rec, mean=0.0, std=rec_weight_scale / np.sqrt(nb_inputs))
        
        self.alpha = alpha
        self.beta = beta
        self.state = None

    def initialize(self, input):
        self.state = self.LIFstate(syn = torch.zeros_like(input, device=input.device),
                                   mem = torch.zeros_like(input, device=input.device),
                                   S = torch.zeros_like(input, device=input.device))

    def reset(self):
        self.state = None

    def forward(self, input):
        h1 = torch.mm(input, self.weight)

        if self.state is None:
            self.initialize(h1)

        syn = self.state.syn
        mem = self.state.mem
        S = self.state.S

        if self.is_recurrent:
            h1 += torch.mm(S, self.weight_rec)
        
        new_syn = self.alpha * syn + h1
        new_mem = self.beta * mem + new_syn

        mthr = new_mem - 1.0
        out = activation(mthr)
        rst = out.detach()

        self.state = self.LIFstate(syn = new_syn,
                                   mem = new_mem * (1.0 - rst),
                                   S = out)

        return out