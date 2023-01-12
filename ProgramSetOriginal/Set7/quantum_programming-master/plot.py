# Rough util to plot data

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# simon_x = [1, 2, 3, 4, 5, 6]
# simon_data = [  0.0000007152557373046,
#                 0.0047228336334228516,
#                 0.010866880416870117,
#                 0.03987264633178711,
#                 0.4229238033294678,
#                 24.408594131469727]
# grover_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# grover_data = [ 0.004805326461791992,
#                 0.004302024841308594,
#                 0.007644176483154297,
#                 0.01414179801940918,
#                 0.029474973678588867,
#                 0.045203208923339844,
#                 0.09999322891235352,
#                 0.29058408737182617,
#                 1.608902931213379,
#                 12.953195095062256,
#                 94.55311107635498]


grover_x = [1, 2, 3, 4, 5]
grover_data = [0.13298274716362357,
                0.1526937591843307,
                0.831627911888063,
                4.18300806125626,
                15.214854561723769
                ]
BV_x = [1, 2, 3, 4]
BV_data = [ 0.12652557669207454,
            0.39602109836414456,
            2.1609569410793483,
            12.760357229970396
]
DJ_x = [1, 2, 3, 4, 5]
DJ_data = [ 0.10980075690895319,
            0.123305250890553,
            0.16017387714236975,
            0.2885974757373333,
            0.9290440510958433
]

grover_total_bits = 12.0 * 10.0
grover_sim_error = [4, 0, 1, 0, 0]
grover_real_error = [11, 8, 25, 35, 36]
grover_sim_error = [x / grover_total_bits for x in grover_sim_error]
grover_real_error = [x / grover_total_bits for x in grover_real_error]

# Data for plotting
fig, ax = plt.subplots()
# ax.plot(DJ_x, DJ_data)
# ax.plot(BV_x, BV_data)
# ax.plot(grover_x, grover_data)
ax.plot(grover_x, grover_sim_error, label='QVM Error')
ax.plot(grover_x, grover_real_error, label='QPU Error')

# ax.set(xlabel='Number of Qubits (n)', ylabel='Time (seconds)',
#     #    title='Deutsch-Jozsa Algorithm')
#        title='Bernstein-Vazirani Algorithm')
#     #    title='Grovers Algorithm')
# # ax.grid()

ax.legend()
ax.set(xlabel='Number of Qubits (n)', ylabel='Error Rate',
       title='Grover Error Rate')

# fig.savefig("DJ.png")
# fig.savefig("BV.png")
# fig.savefig("grover.png")
fig.savefig("grover_error.png")
plt.show()