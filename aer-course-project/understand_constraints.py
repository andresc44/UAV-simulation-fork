input_min = 0.02816169
input_max = 0.14834145

lower_bounds = [-3.5, -3.5, -0.1]
upper_bounds = [3.5, 3.5, 2. ]

#input
Applied_action =  [0.02816169, 0.07833306, 0.14764518, 0.08598955]

#state
Observation = [-5.49738120e-01,  2.15634054e-01, -3.14955812e+00]

c_value_input = [input_min-Applied_action[0], Applied_action[0]-input_max,
                 input_min-Applied_action[1], Applied_action[1]-input_max,
                 input_min-Applied_action[2], Applied_action[2]-input_max,
                 input_min-Applied_action[3], Applied_action[3]-input_max,
                 ]
c_value_state = [lower_bounds[0]-Observation[0], Observation[0]-upper_bounds[0],
                 lower_bounds[1]-Observation[1], Observation[1]-upper_bounds[1],
                 lower_bounds[2]-Observation[2], Observation[2]-upper_bounds[2],
                 ]

c_value = c_value_input.append(c_value_state)

#If all constraints are satisfied, all elements of c_value are negative