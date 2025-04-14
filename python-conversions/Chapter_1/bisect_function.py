def fval(u):
    """
    Evaluate the function for which we want to find a root.

    In this example, we have:
        f(u) = 1 - 2*u

    You can modify this function to change the equation.
    """
    return 1 - 2 * u


def bisection_method(a, b, f, iterations=20):
    """
    Find a root of the function f(u) in the interval [a, b] using the bisection method.

    Parameters:
        a (float): The lower bound of the interval.
        b (float): The upper bound of the interval.
        f (function): The function for which you want to find the root.
        iterations (int): How many times to iterate the bisection process. More iterations give a more precise result.

    Returns:
        float: An approximation of the root of the function.

    Note:
        This function assumes that f(a) and f(b) have opposite signs, i.e., f(a)*f(b) < 0.
        If this is not the case, the bisection method might not find the root.
    """
    # Initialize the lower and upper limits for the search
    lower = a
    fl = f(lower)  # f evaluated at the lower bound
    upper = b
    fu = f(upper)  # f evaluated at the upper bound

    # Main loop: perform the bisection process for a fixed number of iterations
    for i in range(iterations):
        # Calculate the midpoint of the current interval
        mid = (lower + upper) / 2.0
        fm = f(mid)  # Evaluate the function at the midpoint

        # Check which subinterval contains the root:
        # If f(mid) and f(lower) have the same sign, then the root must be in [mid, upper].
        # Otherwise, it is in [lower, mid].
        if fm * fl > 0:
            # f(mid) and f(lower) are both positive or both negative.
            # So, update the lower bound to mid.
            lower = mid
            fl = fm
        else:
            # Otherwise, update the upper bound to mid.
            upper = mid
            fu = fm

        # Optionally, you can print the current bounds to see the progress:
        # print(f"Iteration {i+1}: lower={lower}, upper={upper}, mid={mid}, f(mid)={fm}")

    # After completing the iterations, mid is our best approximation for the root.
    return mid


# Main section of the code that runs when the script is executed directly
if __name__ == "__main__":
    # Set the initial interval for the bisection method
    a = 0  # Lower bound guess
    b = 10  # Upper bound guess

    # Compute the root by calling our bisection_method function with fval as the function to evaluate
    root = bisection_method(a, b, fval, iterations=20)

    # Print the computed root
    print("Computed root:", root)
