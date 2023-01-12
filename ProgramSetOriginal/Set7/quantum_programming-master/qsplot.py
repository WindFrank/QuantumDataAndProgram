# Rough util to plot data

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

simon_data = [ 0.131,
                0.106,
                0.108,
                0.12,
                0.166,
                0.092,
                0.143,
                0.131,
                0.183,
                0.456,
                2.01,
                6.803,
                23.684,
                131.137]

simon_x = range(1,len(simon_data)+1)

# Data for plotting
fig, ax = plt.subplots()
# ax.plot(DJ_x, DJ_data)
ax.plot(simon_x, simon_data)
# ax.plot(grover_x, grover_data)

ax.set(xlabel='Number of Qubits (n)', ylabel='Execution Time (seconds)',
    #    title='Deutsch-Jozsa Algorithm')
       title='Simon\'s Algorithm Execution Time with Increasing n')
    #    title='Grovers Algorithm')
# ax.grid()

# fig.savefig("DJ.png")
fig.savefig("Simon.png")
# fig.savefig("grover.png")
plt.show()

grover_data = [ 0.131,
                0.106,
                0.108,
                0.12,
                0.166,
                0.092,
                0.143,
                0.131,
                0.183,
                0.456,
                2.01,
                6.803,
                23.684,
                131.137]

grover_x = range(1,len(simon_data)+1)

# Data for plotting
fig, ax = plt.subplots()
# ax.plot(DJ_x, DJ_data)
ax.plot(simon_x, simon_data)
# ax.plot(grover_x, grover_data)

ax.set(xlabel='Number of Qubits (n)', ylabel='Execution Time (seconds)',
    #    title='Deutsch-Jozsa Algorithm')
       title='Simon\'s Algorithm Execution Time with Increasing n')
    #    title='Grovers Algorithm')
# ax.grid()

# fig.savefig("DJ.png")
fig.savefig("Simon.png")
# fig.savefig("grover.png")
plt.show()