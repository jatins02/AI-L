import math
from engine import Value
import numpy as np

# what is a toy surface?
# These are well understood 2d or 3d mathematical functions used to benchmark algorithms.
# Because they can be visualised as valleys, peaks and bowls, they are used to check if an optimiser
# get stuck or finds a minimum.

# some classic toy surfaces are:
# The Simple Bowl (Paraboloid): f(x, y) = x^2 + y^2. A perfectly symmetrical bowl. 
# Every optimizer should solve this instantly.

# The Banana Function (Rosenbrock): f(x, y) = (1 - x)^2 + 100(y - x^2)^2. 
# A long, narrow, curved valley. It's notorious because finding the valley is easy, 
# but converging to the global minimum at (1,1) is incredibly difficult.

# The Saddle Point: f(x, y) = x^2 - y^2. Looks like a horse saddle. 
# Along the x-axis, it curves up; along the y-axis, it curves down. 
# Standard gradient descent can get completely trapped or stall out right at the center 
# (0,0) where the slope is flat.



"DEFINING THE TOY SURFACES"
def simple_bowl(x, y):
    return x**2 + y**2

def saddle_point(x, y):
    return x**2 - y**2

def rosenbrock_banana(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2


"DEFINING THE OPTIMISERS"
def run_optimization(surface_fn, start_x, start_y, method="sgd", steps=50, lr=0.01):
    # Initialize the starting positions as Value objects
    x = Value(start_x, label="x")
    y = Value(start_y, label="y")
    
    # Momentum / Adam state tracking variables
    v_x, v_y = 0.0, 0.0  # Used for Momentum and Adam (velocity / 1st moment)
    s_x, s_y = 0.0, 0.0  # Used for Adam (squared gradients / 2nd moment)
    
    # Adam hyperparameters
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    
    print(f"\n--- Optimizing using {method.upper()} ---")
    
    for t in range(1, steps + 1):
        # Forward Pass: Calculate the height (z) on the surface, that is the output of the surface
        loss = surface_fn(x, y)
        
        # Reset historical gradients before backward pass
        x.grad = 0.0
        y.grad = 0.0
        
        # Backward Pass
        loss.backward()
        
        # Capture current gradients
        g_x = x.grad
        g_y = y.grad
        
        # Apply Parameter Updates
        if method == "sgd":
            x.data -= lr * g_x
            y.data -= lr * g_y
            
        elif method == "momentum":
            # beta acts as friction (e.g., 0.9)
            beta = 0.9
            v_x = beta * v_x + (1 - beta) * g_x
            v_y = beta * v_y + (1 - beta) * g_y
            x.data -= lr * v_x
            y.data -= lr * v_y
            
        elif method == "adam":
            # Step 1: Track moving averages of gradients (Moments)
            v_x = beta1 * v_x + (1 - beta1) * g_x
            v_y = beta1 * v_y + (1 - beta1) * g_y
            
            s_x = beta2 * s_x + (1 - beta2) * (g_x ** 2)
            s_y = beta2 * s_y + (1 - beta2) * (g_y ** 2)
            
            # Step 2: Bias Correction
            v_x_corrected = v_x / (1 - beta1 ** t)
            v_y_corrected = v_y / (1 - beta1 ** t)
            s_x_corrected = s_x / (1 - beta2 ** t)
            s_y_corrected = s_y / (1 - beta2 ** t)
            
            # Step 3: Update Data
            x.data -= lr * v_x_corrected / (math.sqrt(s_x_corrected) + eps)
            y.data -= lr * v_y_corrected / (math.sqrt(s_y_corrected) + eps)

        # Print progress every few steps
        if t % 10 == 0 or t == 1:
            print(f"Step {t:02d} | Position: ({x.data:.4f}, {y.data:.4f}) | Height (Loss): {loss.data:.4f}")


"USING THE OPTIMISERS"
if __name__ == "__main__":
    run_optimization(simple_bowl, start_x=4.0, start_y=4.0, method="sgd", steps=50, lr=0.1)
    run_optimization(simple_bowl, start_x=4.0, start_y=4.0, method="momentum", steps=50, lr=0.1)
    run_optimization(simple_bowl, start_x=4.0, start_y=4.0, method="adam", steps=50, lr=0.1)





# INFERENCES:
# pay close attention to how the Height (i.e. the output of surace function) approaches the local minimum
# and the loss approaches zero across different methods.
# - On simple bow, ADAM and Momentum will rapidly outpace vanilla SDG.
# - On saddle_point starting at (0.001, 2.0), vanilla SDG stalls near the flat origin point, while Momentum build 
# speed to roll through it, and Adam adjusts its step size to escape the flat region.




