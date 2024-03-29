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
        out[input >= 0] = 1.0
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
    def __init__(self, nb_inputs, encoder_weight_scale=1.0, bias=0.0, nb_input_copies=1, ds_min=None, ds_max=None):
        super(Encoder, self).__init__()

        enc_gain = torch.empty((nb_inputs,), requires_grad=False)
        enc_bias = torch.empty((nb_inputs,), requires_grad=False)
        torch.nn.init.constant_(enc_gain, encoder_weight_scale)
        torch.nn.init.constant_(enc_bias, bias)
        # torch.nn.init.normal_(enc_gain, mean=0.0, std=encoder_weight_scale)  # TODO update this parameter
        # torch.nn.init.normal_(enc_bias, mean=0.0, std=1.0)

        self.nb_input_copies = nb_input_copies
        self.register_buffer('enc_gain', enc_gain)
        self.register_buffer('enc_bias', enc_bias)
        self.ds_min = ds_min
        self.ds_max = ds_max

    def forward(self, inputs):
        encoder_currents = self.enc_gain * (inputs.tile((self.nb_input_copies,)) + self.enc_bias)
        return encoder_currents

## MN neuron
class MN_neuron_IT(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'i1', 'i2', 'Thr', 'spk'])

    def __init__(self, channels, fanout, params_n, a, A1, A2, b=10, G=50, k1=200, k2=20, gain=1, train=True,
                 dt=1 / 100):
        super(MN_neuron_IT, self).__init__()

        # One-to-one synapse

        self.C = 1

        self.channels = channels
        self.fanout = fanout
        self.params_n = params_n
        self.linear = nn.Parameter(torch.ones(1, fanout), requires_grad=train)

        self.EL = -0.07
        self.Vr = -0.07
        self.R1 = 0.
        self.R2 = 1.
        self.Tr = -0.06
        self.Tinf = -0.05

        # self.b = b  # units of 1/s
        # self.G = G * self.C  # units of 1/s
        self.k1 = 200.  # units of 1/s
        self.k2 = 20.  # units of 1/s

        # self.a = nn.Parameter(torch.tensor(a), requires_grad=True)
        one2N_matrix = torch.ones(1, self.fanout, self.channels, 1)  # shape: 1 (single neuron) x fanout x channels x 1)
        # self.register_buffer('one2N_matrix', torch.ones(1, nb_inputs))

        # shape of a is fanout x channels x params_n
        self.a = torch.permute(nn.Parameter(one2N_matrix * a, requires_grad=train), (0, 1, 3, 2))
        # torch.nn.init.constant_(self.a, a)
        # self.A1 = A1 * self.C
        # self.A2 = A2 * self.C
        self.A1 = torch.permute(nn.Parameter(one2N_matrix * A1 * self.C, requires_grad=train), (0, 1, 3, 2))
        self.A2 = torch.permute(nn.Parameter(one2N_matrix * A2 * self.C, requires_grad=train), (0, 1, 3, 2))
        self.b = torch.permute(nn.Parameter(one2N_matrix * b, requires_grad=train), (0, 1, 3, 2))
        self.G = torch.permute(nn.Parameter(one2N_matrix * G * self.C, requires_grad=train), (0, 1, 3, 2))
        self.k1 = torch.permute(nn.Parameter(one2N_matrix * k1, requires_grad=train), (0, 1, 3, 2))
        self.k2 = torch.permute(nn.Parameter(one2N_matrix * k2, requires_grad=train), (0, 1, 3, 2))
        self.gain = torch.permute(nn.Parameter(one2N_matrix * gain, requires_grad=train), (0, 1, 3, 2))
        self.state = None
        self.dt = dt

    def forward(self, x):
        if self.state is None:
            # channels x fanout x trials
            self.state = self.NeuronState(
                V=torch.ones(x.shape[0], self.fanout, self.params_n, self.channels, device=x.device) * self.EL,
                i1=torch.zeros(x.shape[0], self.fanout, self.params_n, self.channels, device=x.device),
                i2=torch.zeros(x.shape[0], self.fanout, self.params_n, self.channels, device=x.device),
                Thr=torch.ones(x.shape[0], self.fanout, self.params_n, self.channels, device=x.device) * self.Tr,
                spk=torch.zeros(x.shape[0], self.fanout, self.params_n, self.channels, device=x.device))

        V = self.state.V
        i1 = self.state.i1
        i2 = self.state.i2
        Thr = self.state.Thr

        i1 += -self.k1 * i1 * self.dt
        i2 += -self.k2 * i2 * self.dt

        V += self.dt * (self.linear.to(x.device, non_blocking=True) * x * self.gain + i1 + i2 - self.G * (
                V - self.EL)) / self.C

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


## MN neuron
class MN_neuron(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'i1', 'i2', 'Thr', 'spk'])

    def __init__(self, nb_inputs, parameters_combination, dt=1 / 1000, a=5, A1=10, A2=-0.6, b=10, G=50, k1=200, k2=20,
                 R1=0, R2=1, train=True):  # default combination: M2O of the original paper
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

        self.dt = dt  # get dt from sample rate!

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
        V += self.dt * (self.linear.to(x.device, non_blocking=True) * x + i1 + i2 - self.G * (V - self.EL)) / self.C
        Thr += self.dt * (self.a.to(x.device, non_blocking=True) * (V - self.EL) - self.b * (Thr - self.Tinf))

        spk = activation(V - Thr)

        i1 = (1 - spk) * i1 + (spk) * (self.R1 * i1 + self.A1.to(x.device, non_blocking=True))
        i2 = (1 - spk) * i2 + (spk) * (self.R2 * i2 + self.A2.to(x.device, non_blocking=True))
        Thr = (1 - spk) * Thr + (spk) * torch.max(Thr, torch.tensor(self.Tr))
        V = (1 - spk) * V + (spk) * self.Vr

        self.state = self.NeuronState(V=V, i1=i1, i2=i2, Thr=Thr, spk=spk)

        return spk

    def reset(self):
        self.state = None


class IZHI_neuron(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'i', 'spk'])

    def __init__(self, nb_inputs, parameters_combination, dt=1 / 1000, a=0.02, b=0.2, d=8, tau=1, k=150,
                 train=True):  # default combination: M2O of the original paper
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

        self.dt = dt  # get dt from sample rate!

        parameters_list = ["a", "b", "d", "tau", "k"]
        for ii in parameters_list:
            if ii in list(parameters_combination.keys()):
                eval_string = "self.{}".format(ii) + " = " + str(parameters_combination[ii])
                print(eval_string)
                exec(eval_string)

        self.state = None

    def forward(self, x):
        if self.state is None:
            self.state = self.NeuronState(V=torch.ones(x.shape[0], self.N, device=x.device) * self.Vr,
                                          i=torch.zeros(x.shape[0], self.N, device=x.device),
                                          spk=torch.zeros(x.shape[0], self.N, device=x.device))

        V = self.state.V
        i = self.state.i

        V = self.dt * (V + self.tau * (0.04 * (V ** 2) + (5 * V) + 140 - i + x))
        i = self.dt * (i + self.tau * (self.a * ((self.b * V) - i)))

        spk = activation(V - self.Tr)

        i = (1 - spk) * i + (spk) * (i + self.d)
        V = (1 - spk) * V + (spk) * self.Vr

        self.state = self.NeuronState(V=V, i=i, spk=spk)

        return spk

    def reset(self):
        self.state = None


class MN_neuron_sp(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'i1', 'i2', 'Thr', 'spk'])

    def __init__(self, nb_inputs, parameters_combination, dt=1 / 1000, a=5, A1=10, A2=-0.6, b=10, G=50, k1=200, k2=20,
                 R1=0, R2=1, C=1, train=True):  # default combination: M2O of the original paper
        super(MN_neuron_sp, self).__init__()

        # One-to-one synapse
        self.linear = nn.Parameter(torch.ones(1, nb_inputs), requires_grad=False)

        self.C = C

        self.N = nb_inputs

        self.EL = -0.07
        self.Vr = -0.07
        self.Tr = -0.06
        self.Tinf = -0.05

        self.a = a
        self.A1 = A1

        self.A2 = A2
        self.b = b  # units of 1/s
        self.G = G  # units of 1/s
        self.k1 = k1  # units of 1/s
        self.k2 = k2  # units of 1/s
        self.R1 = R1
        self.R2 = R2

        self.dt = dt  # get dt from sample rate!
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
        Thr = (1 - spk) * Thr + (spk) * torch.fmax(Thr, torch.tensor(self.Tr))
        V = (1 - spk) * V + (spk) * self.Vr

        self.state = self.NeuronState(V=V, i1=i1, i2=i2, Thr=Thr, spk=spk)

        return spk

    def reset(self):
        self.state = None


## LIF neuron
class LIF_neuron(nn.Module):
    LIFstate = namedtuple('LIFstate', ['syn', 'mem', 'S'])

    def __init__(self, nb_inputs, nb_outputs, alpha, beta, is_recurrent=True, fwd_weight_scale=1.0,
                 rec_weight_scale=1.0,train=True):
        super(LIF_neuron, self).__init__()

        # weight = torch.empty((nb_inputs, nb_outputs))
        self.weight = torch.nn.Parameter(torch.empty((nb_inputs, nb_outputs)), requires_grad=train)
        torch.nn.init.normal_(self.weight, mean=0.0, std=fwd_weight_scale / np.sqrt(nb_inputs))

        self.is_recurrent = is_recurrent
        if is_recurrent:
            # weight_rec = torch.empty((nb_outputs, nb_outputs))
            # torch.nn.init.normal_(weight_rec, mean=0.0, std=rec_weight_scale / np.sqrt(nb_outputs))
            self.weight_rec = torch.nn.Parameter(torch.empty((nb_outputs, nb_outputs)), requires_grad=train)
            # torch.nn.init.zeros_(self.weight_rec)
            torch.nn.init.normal_(self.weight_rec, mean=0.0, std=rec_weight_scale / np.sqrt(nb_inputs))

        self.alpha = alpha
        self.beta = beta
        self.state = None

    def initialize(self, input):
        self.state = self.LIFstate(syn=torch.zeros_like(input, device=input.device),
                                   mem=torch.zeros_like(input, device=input.device),
                                   S=torch.zeros_like(input, device=input.device))

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

        self.state = self.LIFstate(syn=new_syn,
                                   mem=new_mem * (1.0 - rst),
                                   S=out)

        return out


class ALIF_neuron(nn.Module):
    ALIFstate = namedtuple('ALIFstate', ['syn', 'mem', 'S', 'b'])

    def __init__(self, nb_inputs, nb_outputs, alpha, beta, is_recurrent=True, fwd_weight_scale=1.0,
                 rec_weight_scale=1.0, b_0=1, dt=1, tau_adp=None, beta_adapt=1.8,
                 analog_input=True):
        super(ALIF_neuron, self).__init__()

        self.dt = dt
        # weight = torch.empty((nb_inputs, nb_outputs))
        self.weight = torch.nn.Parameter(torch.empty((nb_inputs, nb_outputs)), requires_grad=True)
        torch.nn.init.normal_(self.weight, mean=0.0, std=fwd_weight_scale / np.sqrt(nb_inputs))

        self.is_recurrent = is_recurrent
        if is_recurrent:
            # weight_rec = torch.empty((nb_outputs, nb_outputs))
            # torch.nn.init.normal_(weight_rec, mean=0.0, std=rec_weight_scale / np.sqrt(nb_outputs))
            self.weight_rec = torch.nn.Parameter(torch.empty((nb_outputs, nb_outputs)), requires_grad=True)
            # torch.nn.init.zeros_(self.weight_rec)
            torch.nn.init.normal_(self.weight_rec, mean=0.0, std=rec_weight_scale / np.sqrt(nb_inputs))

        self.b_0 = b_0  # neural threshold baseline
        self.beta_adapt = beta_adapt
        self.alpha = alpha
        self.beta = beta
        self.ro = float(np.exp(-1. * self.dt / tau_adp))
        self.state = None
        self.analog_input = analog_input
        self.new_mem = None

    def initialize(self, input):
        self.state = self.ALIFstate(syn=torch.zeros_like(input, device=input.device),
                                    mem=torch.zeros_like(input, device=input.device),
                                    S=torch.zeros_like(input, device=input.device),
                                    b= torch.zeros_like(input, device=input.device))

    def reset(self):
        self.state = None

    def forward(self, input):
        # Input = analog current
        # print(input.shape)
        # print(self.weight.shape)
        # print(torch.type(input))
        h1 = torch.mm(input, self.weight.double())

        if self.state is None:
            self.initialize(h1)

        syn = self.state.syn
        mem = self.state.mem
        S = self.state.S

        if self.is_recurrent:
            h1 += torch.mm(S, self.weight_rec)

        # b = self.ro * self.state.b + (1 - self.ro) * S  # decaying factor
        self.b_dec = self.ro * self.state.b
        self.b_update = (1 - self.ro) * S
        b = self.b_dec + self.b_update
        self.thr = self.b_0 + self.beta_adapt * b  # updating threshold (increases with spike, else it decays exp)

        if self.analog_input:
            # Input current
            new_syn = input
        else:
            # Input spikes
            new_syn = self.alpha * syn + h1

        self.new_mem = self.beta * mem + new_syn

        mthr = self.new_mem - self.thr
        out = activation(mthr)
        rst = out.detach()

        self.state = self.ALIFstate(syn=new_syn,
                                    mem=self.new_mem * (1 - rst),
                                    S=out,
                                    b=b)

        return out


class AdexLIF(nn.Module):
    AdexLIFstate = namedtuple('AdexLIFstate', ['V', 'W', 'S'])

    def __init__(self, n_in, n_out, params_n, channels, dt=1.):
        super(AdexLIF, self).__init__()

        self.linear = nn.Parameter(torch.ones(1, n_out), requires_grad=True)
        self.dt = dt * 1000
        self.n_in = n_in
        self.n_out = n_out
        self.params_n = params_n
        self.channels = channels

        ## If want to use as a parameter use nn.Parameter(...)
        self.Vr = nn.Parameter(torch.tensor(-70.), requires_grad=True)  # resting potential
        self.Vth = nn.Parameter(torch.tensor(-30.), requires_grad=True)  # Firing threshold
        self.Vrh = nn.Parameter(torch.tensor(-50.), requires_grad=True)
        self.Vreset = nn.Parameter(torch.tensor(-51.), requires_grad=True)  # reset potential
        self.delta_T = nn.Parameter(torch.tensor(2.), requires_grad=True)  # Sharpness of the exponential term
        self.a = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # Adaptation-Voltage coupling
        self.b = nn.Parameter(torch.tensor(7.0), requires_grad=True)  # Spike-triggered adaptation current
        self.R = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # Resistance
        self.taum = nn.Parameter(torch.tensor(50.), requires_grad=True)  # membrane time scale
        self.tauw = nn.Parameter(torch.tensor(1000.), requires_grad=True)  # Adaptation time constant

        self.state = None

    def reset(self):
        self.state = None
        # print('reset done')

    def forward(self, input):
        if self.state is None:
            self.state = self.AdexLIFstate(
                V=torch.zeros(input.shape[0], self.n_out, self.params_n, self.channels, device=input.device) + self.Vr,
                W=torch.zeros(input.shape[0], self.n_out, self.params_n, self.channels, device=input.device),
                S=torch.zeros(input.shape[0], self.n_out, self.params_n, self.channels, device=input.device))
        # print(input.shape[0])
        V = self.state.V
        W = self.state.W
        I = (self.linear * input)
        dV = (-(V - self.Vr) + self.delta_T * torch.exp((V - self.Vrh) / self.delta_T) + self.R * (I - W)) / (self.taum)
        dW = (self.a * (V - self.Vr) - W) / self.tauw

        V = V + self.dt * dV
        # print('input_max',input.max())
        # print('I_max',I.max())
        # print('dV_max',dV.max())
        W = W + self.dt * dW

        spk = activation(V - self.Vth)
        # spk = ((V - self.Vth) > 0).float()

        W = (1 - spk) * W + (spk) * (W + self.b)
        V = (1 - spk) * V + (spk) * self.Vreset

        self.state = self.AdexLIFstate(V=V, W=W, S=spk)
        # print('spk_sum',spk.sum())
        # print('spk_max',spk.max())
        return spk
