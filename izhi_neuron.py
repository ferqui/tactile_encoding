'''
Copyright (C) 2021
Authors: Alejandro Pequeno-Zurro

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.
'''
# import os
import numpy as np
# import matplotlib.pyplot as plt
# import time
from dataclasses import dataclass

@dataclass
class IzhParameters:
    a: float
    b: float
    c: int
    d: int
    tau: int
    k: int

sa_params = IzhParameters(a=0.02, b=0.2, c=-65, d=8, tau=1, k=150)
ra_params = IzhParameters(a=0.02, b=0.25, c=-65, d=8, tau=0.25, k=7.5)

def gen_spikes(A, scale, params):
    ''' Convert data matrix A into spiking output using the Izhikevich neuron model.
        A: Data matrix where each row of A is a sensor and each column is a sample.
           The data is assumed to be sampled at 1 kHz.
        scale: gain on sensor output
        params: Izhikevich neuron model parameters.
        Returns matrices of the analog output voltage (v), the recovery variable (u),
                and a binary spiking output (spikes)
    '''
    num_sens = len(A)
    num_samp = len(A[0])

    v0 = -65
    v = np.empty([num_sens, num_samp+1])
    u0 = v0 * params.b
    u = np.empty([num_sens, num_samp+1])
    spikes = np.zeros([num_sens, num_samp+1],dtype=bool)

    for i in range(num_sens):
        for t in range(num_samp + 1):
            if t == 0:
                v[i][t] = v0
                u[i][t] = u0
            else:
                I = A[i][t-1] * scale
                if v[i][t-1] == 30:
                    v_prev = params.c
                else:
                    v_prev = v[i][t-1]
                u_prev = u[i][t-1]
                v[i][t] = v_prev + params.tau * (0.04 * (v_prev**2) +\
                    (5 * v_prev) + 140 - u_prev + I)
                u[i][t] = u_prev + params.tau *\
                    (params.a * ((params.b * v[i][t]) - u_prev))
                if v[i][t] >= 30:
                    v[i][t] = 30
                    u[i][t] = u[i][t] + params.d
                    spikes[i][t] = 1

    neuron, times = np.nonzero(spikes)
    sparse_spikes = [times[neuron == curr_neuron] for curr_neuron in range(len(A))]
    return sparse_spikes
