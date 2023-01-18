# set up neuron parameters and input current
class_names = [
    "Tonic spiking",
    "Class 1",
    "Spike frequency adaptation",
    "Phasic spiking",
    "Accommodation",
    "Threshold variability",
    "Rebound spike",
    "Class 2",
    "Integrator",
    "Input bistability",
    "Hyperpolarizing spiking",
    "Hyperpolarizing bursting",
    "Tonic bursting",
    "Phasic bursting",
    "Rebound burst",
    "Mixed mode",
    "Afterpotentials",
    "Basal bistability",
    "Preferred frequency",
    "Spike latency",
]


input_currents = {
    "Tonic spiking": 1.5,
    "Class 1": 1.000001, # 1 + 1E-6
    "Spike frequency adaptation": 2,
    "Phasic spiking": 1.5,
    "Accommodation": [1.5, 0, 0.5, 1, 1.5, 0],
    "Threshold variability": [1.5, 0, -1.5, 0, 1.5, 0],
    "Rebound spike": [0, -3.5, 0],
    "Class 2": 2.000002, # 2(1 + 1E-6)
    "Integrator": [1.5, 0, 1.5, 0, 1.5, 0, 1.5, 0],
    "Input bistability": [1.5, 1.7, 1.5, 1.7],
    "Hyperpolarizing spiking": -1,
    "Hyperpolarizing bursting": -1,
    "Tonic bursting": 2,
    "Phasic bursting": 1.5,
    "Rebound burst": [0, -3.5, 0],
    "Mixed mode": 2,
    "Afterpotentials": [2, 0],
    "Basal bistability": [5, 0, 5, 0],
    "Preferred frequency": [5, 0, 4, 0, 5, 0, 4, 0],
    "Spike latency": [8, 0],
}


neuron_parameters = []

tonic_spiking = {
    "a": 0,
    "A1": 0,
    "A2": 0,
}
neuron_parameters.append(tonic_spiking)

class1 = {
    "a": 0,
    "A1": 0,
    "A2": 0,
}
neuron_parameters.append(class1)


spike_frequency_adaptation = {
    "a": 5,
    "A1": 0,
    "A2": 0,
}
neuron_parameters.append(spike_frequency_adaptation)

phasic_spiking = {
    "a": 5,
    "A1": 0,
    "A2": 0,
}
neuron_parameters.append(phasic_spiking)

accommodation = {
    "a": 5,
    "A1": 0,
    "A2": 0,
}
neuron_parameters.append(accommodation)

threshold_variability = {
    "a": 5,
    "A1": 0,
    "A2": 0,
}
neuron_parameters.append(threshold_variability)

threshold_variability = {
    "a": 5,
    "A1": 0,
    "A2": 0,
}
neuron_parameters.append(threshold_variability)

rebound_spike = {
    "a": 5,
    "A1": 0,
    "A2": 0,
}
neuron_parameters.append(rebound_spike)

class2 = {
    "a": 5,
    "A1": 0,
    "A2": 0,
}
neuron_parameters.append(class2)

integrator = {
    "a": 5,
    "A1": 0,
    "A2": 0,
}
neuron_parameters.append(integrator)

input_bistability = {
    "a": 5,
    "A1": 0,
    "A2": 0,
}
neuron_parameters.append(input_bistability)


hyperpolarizing_spiking = {
    "a": 30,
    "A1": 0,
    "A2": 0,
}
neuron_parameters.append(hyperpolarizing_spiking)


hyperpolarizing_bursting = {
    "a": 30,
    "A1": 10,
    "A2": -0.6,
}
neuron_parameters.append(hyperpolarizing_bursting)


tonic_bursting = {
    "a": 5,
    "A1": 10,
    "A2": -0.6,
}
neuron_parameters.append(tonic_bursting)

phasic_bursting = {
    "a": 5,
    "A1": 10,
    "A2": -0.6,
}
neuron_parameters.append(phasic_bursting)

rebound_burst = {
    "a": 5,
    "A1": 10,
    "A2": -0.6,
}
neuron_parameters.append(rebound_burst)


mixed_mode = {
    "a": 5,
    "A1": 5,
    "A2": -0.3,
}
neuron_parameters.append(mixed_mode)

afterpotentials = {
    "a": 5,
    "A1": 5,
    "A2": -0.3,
}
neuron_parameters.append(afterpotentials)


basal_bistability = {
    "a": 0,
    "A1": 8,
    "A2": -0.1,
}
neuron_parameters.append(basal_bistability)


preferred_frequency = {
    "a": 5,
    "A1": -3,
    "A2": 0.5,
}
neuron_parameters.append(preferred_frequency)


spike_latency = {
    "a": 5,
    "A1": -3,
    "A2": 0.5,
}
neuron_parameters.append(spike_latency)
