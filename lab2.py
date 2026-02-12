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
from scipy.spatial.distance import pdist, squareform

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
# end

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
# end

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
The red line stays in its starting hole at x = -2 due to the gradient step relying on the slope
of the terrain. As the hole the slope is near zero. even as it moves towards the edge of the hole,
the slope angles it back into the hole thus keeping the particle trapped in the hole with this method.
The blue line manages to escape since it randomly pick an x value to jump to, instead of relying on the
slope. This random picker prefer areas of lower y values especially as the simulation goes on since
the probability of it jumping out of a hole decreases as the Temperature cools. Therefore the blue line
tends towards the middle hole at x = 0 and ends up staying their due to cooling.
'''


# - Part 2 - #
# Crystallization of Argon

# Excercise 2.1 
def Lennard_Jones_Energy(positions):
    """
    Calculates total energy of N atoms.
    positions: An array of shape (N, 2) containing (x, y) for each atom.
    """
    N = len(positions)
    total_energy = 0

    # Loop over every unique pair of atoms
    for i in range(N):
        for j in range(i + 1, N):
            # Distance formula
            dist_vector = positions[i] - positions[j]
            r = np.linalg.norm(dist_vector) 

            # Prevent division by zero if atoms overlap perfectly
            if r < 0.01:
                print(r)
                r = 0.01

            # Lennard-Jones formula (epsilon=1, sigma=1)
            # Term 1: Repulsion (r^-12)
            # Term 2: Attraction (r^-6)
            energy = 4 * ((1/r)**12 - (1/r)**6)
            total_energy += energy

    return total_energy
# end

# Exercise 2.2 Annealing the Crystal

def Solve_Cluster(N_atoms, steps=5000):
    # 1. Initialize random positions
    positions = np.random.uniform(-2, 2, size=(N_atoms, 2))
    curr_E = Lennard_Jones_Energy(positions)

    T = 5.0
    cooling_rate = 0.999

    history_E = []
    size = 2
    for i in range(steps):
        #size = 0.04
        # 2. Propose a move
        new_positions = positions.copy()
        index = np.random.randint(N_atoms)
        dx = np.random.uniform(-size, size)
        dy = np.random.uniform(-size, size)
        new_positions[index][0] = dx
        new_positions[index][1] = dy

        # 3. Calculate Energy Change
        new_E = Lennard_Jones_Energy(new_positions)
        delta_E = new_E - curr_E

        # 4. Metropolis Criterion
        accept = False

        if (delta_E < 0):
            accept = True
        elif (delta_E > 0):    # if delta_E = 0  => accept = False
            prob = np.exp(-delta_E/T)
            if(np.random.rand() <= prob):
                accept = True

        if accept:
            positions = new_positions
            curr_E = new_E

        T *= cooling_rate
        history_E.append(curr_E)

    return positions, history_E
# end

# --- RUN THE EXPERIMENT ---
N = 4 # Try N=3 (triangle), N=4 (diamond), N=7 (hexagon with center)
final_pos, energy_log = Solve_Cluster(N)

# Plotting
plt.figure(figsize=(12, 5))

# Plot 1: The Energy Drop (Cooling)
plt.subplot(1, 2, 1)
plt.plot(energy_log)
plt.title(f"System Cooling (Final E: {energy_log[-1]:.2f})")
plt.xlabel("Time Step")
plt.ylabel("Total Energy")

# Plot 2: The Final Crystal Shape
plt.subplot(1, 2, 2)
plt.scatter(final_pos[:,0], final_pos[:,1], s=500, c='cyan', edgecolors='black', zorder=2)

# Draw bonds between close atoms to visualize structure
dists = squareform(pdist(final_pos))
for i in range(N):
    for j in range(i+1, N):
        if dists[i,j] < 1.5: # If close enough to bond
            plt.plot([final_pos[i,0], final_pos[j,0]], [final_pos[i,1], final_pos[j,1]], 'k-', lw=2, zorder=1)

plt.title(f"Final Geometry (N={N})")
plt.axis('equal')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Question 2
# Run the code for N=4. What shape does it form? Does it ever form a square? Why or why not? 
'''
For N = 4 the simulation never produces a square, it is always two sets of triangles which form a diamond.
These two triangles tend to be equilateral as well. With the pyramid structure 5 of the connecting lines
are all of the same length and only the line that would connect the two tips is the longest. With a square
only 4 of the lines would be the same length, leaving the two diagonals to be longer lengths. I believe that
it is this property which makes it so it is always a diamond and not a square for these crystal structures.
'''
