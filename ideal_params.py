"""
Sets up neuron parameters and input currents
"""

input_currents = {
    "Tonic spiking": [1.5],
    "Class 1": [1.000001],  # 1 + 1E-6
    "Spike frequency adaptation": [2],
    "Phasic spiking": [1.5],
    "Accommodation": [1.5, 0, 0.5, 1, 1.5, 0],
    "Threshold variability": [1.5, 0, -1.5, 0, 1.5, 0],
    "Rebound spike": [0, -3.5, 0],
    "Class 2": [2.000002],  # 2(1 + 1E-6)
    "Integrator": [1.5, 0, 1.5, 0, 1.5, 0, 1.5, 0],
    "Input bistability": [1.5, 1.7, 1.5, 1.7],
    "Hyperpolarizing spiking": [-1],
    "Hyperpolarizing bursting": [-1],
    "Tonic bursting": [2],
    "Phasic bursting": [1.5],
    "Rebound burst": [0, -3.5, 0],
    "Mixed mode": [2],
    "Afterpotentials": [2, 0],
    "Basal bistability": [5, 0, 5, 0],
    "Preferred frequency": [5, 0, 4, 0, 5, 0, 4, 0],
    "Spike latency": [8, 0],
}


time_points = {
    "Accommodation": [100, 600, 700, 800, 900], # 1000 ms
    "Threshold variability": [40, 200, 220, 260, 280], # 400 ms
    "Rebound spike": [50, 800], # 1000 ms
    "Integrator": [20, 30, 50, 300, 320, 350, 370], # 400 ms
    "Input bistability": [100, 500, 600], # 1000 ms
    "Rebound burst": [100, 600], # 1000 ms
    "Afterpotentials": [15], # 200 ms
    "Basal bistability": [10, 100, 110], # 200 ms
    "Preferred frequency": [10, 20, 30, 400, 410, 450, 460], # 800 ms
    "Spike latency": [2], # 50 ms
}

runtime = { # ms
    "Tonic spiking": 200,
    "Class 1": 500,
    "Spike frequency adaptation": 200,
    "Phasic spiking": 500,
    "Accommodation": 1000,
    "Threshold variability": 400,
    "Rebound spike": 1000,
    "Class 2": 300,
    "Integrator": 400,
    "Input bistability": 1000,
    "Hyperpolarizing spiking": 400,
    "Hyperpolarizing bursting": 400,
    "Tonic bursting": 500,
    "Phasic bursting": 500,
    "Rebound burst": 1000,
    "Mixed mode": 500,
    "Afterpotentials": 200,
    "Basal bistability": 200,
    "Preferred frequency": 800,
    "Spike latency": 50,
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
