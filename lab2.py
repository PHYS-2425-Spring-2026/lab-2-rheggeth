#####################################
#                                   #
#             Rheggeth              #
#       Last Edit: 27/01/26         #
#  Lab 4: Stat. Mech. Optimization  #
#                                   #
#####################################

# imports
import numpy as np
import matplotlib.pyplot as plt

# - Part 1 - #

# 1D energy function
def Energy_Function(x):
    # the potential energy landscape
    return x**2 - 4 * np.cos(4 * np.pi * x)

# what the landsacep looks like
x_vals = np.linspace(-3, 3, 1000)
plt.figure(figsize=(10, 5))
plt.plot(x_vals, Energy_Function(x_vals), "k-", alpha=0.6, label="Energy Surface")
plt.title("The Challenge: Many Local Minima")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Excercise 1.1 Gradient Descent #

def Run_Gradient_Descent(x_start, steps=50, learning_rate=0.005):
    path = [x_start]
    x = x_start

    for i in range(steps):
        # Numerical Derivative (Slope)
        dx = 0.001
        slope = (Energy_Function(x + dx) - Energy_Function(x - dx)) / (2*dx)
    
        # TODO: Update x using the Gradient Descent rule
        x = path[-1] - learning_rate*slope
        path.append(x)

    return np.array(path)

# Start at x = -2 (A bad spot!)
x_start = -2.0
gd_path = Run_Gradient_Descent(x_start)

print(f"Gradient Descent started at {x_start} and finished at {gd_path[-1]:.3f}")
print(f"True Global Minimum is at 0.0")

# Excercise 1.2 Simulated Annealing

def Run_Annealing(x_start, T_start=10.0, cooling_rate=0.99, steps=1000):
    x = x_start
    E = Energy_Function(x)
    T = T_start

    path = [x]

    for i in range(steps):
        # 1. Propose a random small move
        # sets x_new to a range out of bounds
        x_new = -5
        # confines x_new to be within our landcape: [-3, 3]
        while(-3 > x_new or x_new > 3):
            dx = np.random.uniform(-0.5, 0.5)
            x_new = x + dx

        # 2. Calculate Energy Change
        E_new = Energy_Function(x_new)
        delta_E = E_new - E

        # 3. Metropolis Logic
        # TODO: Decide whether to accept the move
        accept = False

        if (delta_E < 0):
            accept = True
        elif (delta_E > 0):    # if delta_E = 0  => accept = False
            prob = np.exp(-delta_E/T)
            if(np.random.rand() <= prob):
                accept = True

        if accept:
            x = x_new
            E = E_new

        # 4. Cool down
        T = T * cooling_rate
        path.append(x)

    return np.array(path)

# --- VISUALIZATION ---
# We compare the two methods side-by-side

# 1. Run Gradient Descent
gd_path = Run_Gradient_Descent(-2.0, steps=50, learning_rate=0.005)

# 2. Run Simulated Annealing (Uses solution function for demo)
sa_path = Run_Annealing(-2.0, T_start=10.0, cooling_rate=0.99, steps=1000)

plt.figure(figsize=(12, 6))
plt.plot(x_vals, Energy_Function(x_vals), "k-", alpha=0.3, label="Landscape")

# Plot Gradient Descent (Red)
plt.plot(gd_path, Energy_Function(gd_path), "o-", color="red", label="Gradient Descent", markersize=4)
plt.plot(gd_path[-1], Energy_Function(gd_path[-1]), "rx", markersize=15, markeredgewidth=3, label="GD Final")

# Plot Simulated Annealing (Blue)
# We only plot every 10th step so the graph isn't messy
plt.plot(sa_path[::10], Energy_Function(sa_path)[::10], "o-", color="blue", alpha=0.4, label="Simulated Annealing", markersize=2)
plt.plot(sa_path[-1], Energy_Function(sa_path[-1]), "b*", markersize=20, label="SA Final")

plt.title("Comparison: Greedy vs Thermal")
plt.legend()
plt.show()

# Question 1
# Why does the red line stop in the hole at x≈−2, while the blue line manages to escape to x≈0?
'''
hi
'''

# - Part 2 - #


