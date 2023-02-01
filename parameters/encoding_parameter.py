"""
Parameters are organized as following:
"name", min_val, max_val, step_size, init_val
# TODO include init_val for GUI!
"""
mn_parameter = [
                ["a", -1500, 0, 10],
                ["A1", -15, 5, 0.01],
                ["A2", -1, 8, 0.01],
                ["b", -40, 40, 1],  # units of 1/s (10)
                ["G", 0, 200, 10],  # units of 1/s (50)
                ["k1", 0, 400, 10],  # units of 1/s (200)
                ["k2", 0, 40, 1],  # units of 1/s (20)
                ["R1", 0, 1, 0.5],
                ["R2", 0, 1, 0.5],
            ]

iz_parameter = [
                ["a", -500, 20, 10],
                ["b", -10, 5, 0.01],
                ["d", -2, 2, 0.01],
                ["k", -40, 40, 1],
            ]

lif_parameter = [
                ["placeholder", 1, 10, 5],
            ]