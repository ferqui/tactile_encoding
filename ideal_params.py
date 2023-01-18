# set up neuron parameters and input current
input_currents = {
    "Tonic spiking": 1.5,
    "Class 1": 1.000001,  # 1 + 1E-6
    "Spike frequency adaptation": 2,
    "Phasic spiking": 1.5,
    "Accommodation": [1.5, 0, 0.5, 1, 1.5, 0],
    "Threshold variability": [1.5, 0, -1.5, 0, 1.5, 0],
    "Rebound spike": [0, -3.5, 0],
    "Class 2": 2.000002,  # 2(1 + 1E-6)
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


neuron_parameters = {}

neuron_parameters["Tonic spiking"] = {
    "a": 0,
    "A1": 0,
    "A2": 0,
}

neuron_parameters["Class 1"] = {
    "a": 0,
    "A1": 0,
    "A2": 0,
}


neuron_parameters["Spike frequency adaptation"] = {
    "a": 5,
    "A1": 0,
    "A2": 0,
}

neuron_parameters["Phasic spiking"] = {
    "a": 5,
    "A1": 0,
    "A2": 0,
}

neuron_parameters["Accommodation"] = {
    "a": 5,
    "A1": 0,
    "A2": 0,
}

neuron_parameters["Threshold variability"] = {
    "a": 5,
    "A1": 0,
    "A2": 0,
}

neuron_parameters["Rebound spike"] = {
    "a": 5,
    "A1": 0,
    "A2": 0,
}

neuron_parameters["Class 2"] = {
    "a": 5,
    "A1": 0,
    "A2": 0,
}

neuron_parameters["Integrator"] = {
    "a": 5,
    "A1": 0,
    "A2": 0,
}

neuron_parameters["Input bistability"] = {
    "a": 5,
    "A1": 0,
    "A2": 0,
}


neuron_parameters["Hyperpolarizing spiking"] = {
    "a": 30,
    "A1": 0,
    "A2": 0,
}


neuron_parameters["Hyperpolarizing bursting"] = {
    "a": 30,
    "A1": 10,
    "A2": -0.6,
}


neuron_parameters["Tonic bursting"] = {
    "a": 5,
    "A1": 10,
    "A2": -0.6,
}

neuron_parameters["Phasic bursting"] = {
    "a": 5,
    "A1": 10,
    "A2": -0.6,
}

neuron_parameters["Rebound burst"] = {
    "a": 5,
    "A1": 10,
    "A2": -0.6,
}


neuron_parameters["Mixed mode"] = {
    "a": 5,
    "A1": 5,
    "A2": -0.3,
}

neuron_parameters["Afterpotentials"] = {
    "a": 5,
    "A1": 5,
    "A2": -0.3,
}


neuron_parameters["Basal bistability"] = {
    "a": 0,
    "A1": 8,
    "A2": -0.1,
}


neuron_parameters["Preferred frequency"] = {
    "a": 5,
    "A1": -3,
    "A2": 0.5,
}


neuron_parameters["Spike latency"] = {
    "a": 5,
    "A1": -3,
    "A2": 0.5,
}
