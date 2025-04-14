import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math

# -------------------------------
# Helper functions
# -------------------------------


def set_default_plot_params():
    """Set default plotting parameters similar to MATLAB's set(0,...) command."""
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["lines.linewidth"] = 1.2


def contiguous_segments(indices):
    """
    Given an array of indices, break it up into sublists where indices are contiguous.
    For example, [3,4,5, 7,8] becomes [[3,4,5], [7,8]].
    """
    segments = []
    if len(indices) == 0:
        return segments
    current = [indices[0]]
    for idx in indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            segments.append(current)
            current = [idx]
    segments.append(current)
    return segments


def de_rhs(t, s, A):
    """Right–hand side for the ODE system: s' = A*s."""
    return A @ s


# -------------------------------
# Main plotting function
# -------------------------------


def make_de_plots():
    # Set up default plot parameters
    set_default_plot_params()

    # -------------------------------------------------
    # Figure 1: Plot the right-hand side (rhs) of u' = f(u)
    # -------------------------------------------------
    # Define the x–coordinate range and parameters
    x = np.arange(-0.2, 1.2001, 0.002)
    N = len(x)
    al = 0.25  # parameter alpha
    A_scale = 10  # scale factor (we call it A_scale here to avoid global variables)

    # Define the right-hand side function: f(u) = A*x*(1-x)*(x-al)
    f = A_scale * x * (1 - x) * (x - al)

    # Start Figure 1
    plt.figure(1)
    plt.plot(x, f, label="f(u)")
    plt.plot(x, np.zeros(N), "--", label="0 line")
    # Plot markers at u = 0, al, and 1
    plt.plot([0, al, 1], [0, 0, 0], "*k", markersize=10)

    plt.axis([-0.2, 1.2, -1, 1])
    plt.xticks(np.arange(-0.2, 1.21, 0.2))
    plt.xlabel("u")
    plt.ylabel("du/dt")

    # Add annotation arrows in axes–fraction coordinates (mimicking MATLAB's textarrow)
    # Arrow from (0.35, 0.7) to (0.25, 0.7)
    plt.gca().annotate(
        "",
        xy=(0.25, 0.7),
        xytext=(0.35, 0.7),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    # Arrow from (0.55, 0.7) to (0.65, 0.7)
    plt.gca().annotate(
        "",
        xy=(0.65, 0.7),
        xytext=(0.55, 0.7),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )

    plt.title("Figure 1: f(u) and u-axis")
    plt.legend()

    # -------------------------------------------------
    # Figure 2: Plot the integral F vs x
    # -------------------------------------------------
    # Compute F = (al*log(|1-x|) - log(|x-al|) + (1-al)*log(|x|))/(A*al*(al-1))
    # Use np.abs and np.log; note potential warnings for log(0)
    F = (
        al * np.log(np.abs(1 - x))
        - np.log(np.abs(x - al))
        + (1 - al) * np.log(np.abs(x))
    ) / (A_scale * al * (al - 1))

    plt.figure(2)
    plt.plot(x, F, label="F(u)")
    plt.plot(x, np.zeros(N), "--", label="0 line")
    plt.xlabel("u")
    plt.ylabel("F(u)")
    plt.xticks(np.arange(-0.2, 1.21, 0.2))
    plt.axis([-0.2, 1.2, -3, 2])
    plt.title("Figure 2: Integral F(u) vs u")
    plt.legend()

    # -------------------------------------------------
    # Figure 3: Reverse the axes: plot F on x–axis and u on y–axis
    # -------------------------------------------------
    plt.figure(3)
    plt.plot(F, x, label="u(F)")
    # Add horizontal dashed lines at u = 0, al, and 1
    plt.plot([-3, 3], [al, al], "--", linewidth=2)
    plt.plot([-3, 3], [0, 0], "--", linewidth=2)
    plt.plot([-3, 3], [1, 1], "--", linewidth=2)
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.axis([-3, 2, -0.2, 1.2])
    plt.title("Figure 3: Reversed Axes")
    plt.legend()

    # -------------------------------------------------
    # Figure 4: Nullclines and flow directions for u'' + f(u) = 0
    # -------------------------------------------------
    plt.figure(4)
    # Plot horizontal line along u-axis and vertical lines at 0, al and 1
    plt.plot(x, np.zeros(N), "--", linewidth=2)
    plt.plot([0, 0], [-1, 1], "--", linewidth=2)
    plt.plot([al, al], [-1, 1], "--", linewidth=2)
    plt.plot([1, 1], [-1, 1], "--", linewidth=2)
    # Plot markers at u = 0, al, 1 with zeros on v axis
    plt.plot([0, al, 1], [0, 0, 0], "*", markersize=12)
    plt.axis([-0.2, 1.2, -1, 1])
    plt.xticks(np.arange(-0.2, 1.21, 0.2))

    # Add several annotation arrows (using axes fraction coordinates)
    plt.gca().annotate(
        "",
        xy=(0.215, 0.72),
        xytext=(0.265, 0.72),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    plt.gca().annotate(
        "",
        xy=(0.36, 0.72),
        xytext=(0.41, 0.72),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    plt.gca().annotate(
        "",
        xy=(0.77, 0.72),
        xytext=(0.83, 0.72),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    plt.gca().annotate(
        "",
        xy=(0.265, 0.32),
        xytext=(0.215, 0.32),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    plt.gca().annotate(
        "",
        xy=(0.41, 0.32),
        xytext=(0.36, 0.32),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    plt.gca().annotate(
        "",
        xy=(0.83, 0.32),
        xytext=(0.77, 0.32),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    # Additional arrows on the sides
    plt.gca().annotate(
        "",
        xy=(0.6, 0.56),
        xytext=(0.6, 0.49),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    plt.gca().annotate(
        "",
        xy=(0.3, 0.49),
        xytext=(0.3, 0.56),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    plt.gca().annotate(
        "",
        xy=(0.18, 0.56),
        xytext=(0.18, 0.49),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    plt.gca().annotate(
        "",
        xy=(0.85, 0.49),
        xytext=(0.85, 0.56),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    plt.gca().annotate(
        "",
        xy=(0.28, 0.6),
        xytext=(0.32, 0.63),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    plt.gca().annotate(
        "",
        xy=(0.32, 0.39),
        xytext=(0.28, 0.42),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    plt.gca().annotate(
        "",
        xy=(0.62, 0.42),
        xytext=(0.58, 0.39),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    plt.gca().annotate(
        "",
        xy=(0.58, 0.63),
        xytext=(0.62, 0.6),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    plt.xlabel("u")
    plt.ylabel("v")
    plt.title("Figure 4: Nullclines and Flow Directions")

    # -------------------------------------------------
    # Figure 5: Integral Curves (trajectories)
    # -------------------------------------------------
    # Recompute F differently: F = (A/12)*x^2*(4*al*x - 3*x^2 - 6*al + 4*x)
    F_traj = (A_scale / 12) * x**2 * (4 * al * x - 3 * x**2 - 6 * al + 4 * x)

    # Define initial points for x
    xj = np.array([0, 0.15, 0.6, 1])

    plt.figure(5)
    for xs in xj:
        # Compute the function value at the initial point
        Fj = (A_scale / 12) * xs**2 * (4 * al * xs - 3 * xs**2 - 6 * al + 4 * xs)
        # Compute the difference Fdf = Fj - F_traj
        Fdf = Fj - F_traj
        # Find indices where Fdf is nonnegative (so that the square root is real)
        ndx = np.where(Fdf >= 0)[0]
        if len(ndx) == 0:
            continue  # No valid points to plot
        # Split indices into contiguous segments
        segments = contiguous_segments(ndx)
        for seg in segments:
            x_seg = x[seg]
            Fdiff_seg = Fj - F_traj[seg]
            # To avoid numerical issues, clip negative values (they should be near zero)
            Fdiff_seg = np.clip(Fdiff_seg, 0, None)
            sqrt_term = np.sqrt(Fdiff_seg)
            # Plot both the positive and negative branches
            plt.plot(x_seg, sqrt_term, linewidth=2)
            plt.plot(x_seg, -sqrt_term, linewidth=2)

    # Add additional reference lines and markers
    plt.plot(x, np.zeros(N), "--", linewidth=2)
    plt.plot([0, 0], [-1, 1], "--", linewidth=2)
    plt.plot([al, al], [-1, 1], "--", linewidth=2)
    plt.plot([1, 1], [-1, 1], "--", linewidth=2)
    plt.plot([0, al, 1], [0, 0, 0], "*", markersize=12)
    plt.axis([-0.2, 1.2, -1, 1])
    plt.xticks(np.arange(-0.2, 1.21, 0.2))
    plt.xlabel("u")
    plt.ylabel("v")
    plt.title("Figure 5: Integral Curves (Trajectories)")
    # Sample annotations for figure 5 – here we add two arrows
    plt.gca().annotate(
        "",
        xy=(0.48, 0.33),
        xytext=(0.53, 0.36),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    plt.gca().annotate(
        "",
        xy=(0.53, 0.68),
        xytext=(0.48, 0.72),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )

    # -------------------------------------------------
    # Phase Portraits: Figures 6, 7, 8, and 9
    # -------------------------------------------------
    tstep = 0.01  # time step size

    # ---- Figure 6: Saddle point dynamics ----
    # ODE: s' = A*s with A = [[0, 1], [1, 0]]
    A1 = np.array([[0, 1], [1, 0]])
    s0 = np.array([-1.13, 1.1])
    tspan = (0, 5)
    t_eval = np.arange(tspan[0], tspan[1] + tstep, tstep)
    sol = solve_ivp(
        lambda t, s: de_rhs(t, s, A1), tspan, s0, method="BDF", t_eval=t_eval
    )
    plt.figure(6)
    # Plot the solution and its symmetric images:
    plt.plot(sol.y[0, :], sol.y[1, :], linewidth=2)
    plt.plot(-sol.y[0, :], -sol.y[1, :], linewidth=2)
    plt.plot(sol.y[1, :], sol.y[0, :], linewidth=2)
    plt.plot(-sol.y[1, :], -sol.y[0, :], linewidth=2)
    plt.xlabel("u")
    plt.ylabel("v")
    # Additionally, plot the stable/unstable manifolds:
    plt.plot([0, 1], [0, 1], linewidth=2)
    plt.plot([0, 1], [0, -1], linewidth=2)
    plt.plot([0, -1], [0, 1], linewidth=2)
    plt.plot([0, -1], [0, -1], linewidth=2)
    # Plot nullclines for reference:
    plt.plot([-5, 5], [0, 0], "--", linewidth=2)
    plt.plot([0, 0], [-5, 5], "--", linewidth=2)
    plt.plot(0, 0, "*", markersize=12)
    plt.axis([-1, 1, -1, 1])
    # Some annotation arrows (axes fraction coordinates)
    plt.gca().annotate(
        "",
        xy=(0.3, 0.49),
        xytext=(0.3, 0.56),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    plt.gca().annotate(
        "",
        xy=(0.75, 0.56),
        xytext=(0.75, 0.49),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    plt.title("Figure 6: Saddle Point Phase Portrait")

    # ---- Figure 7: Unstable spiral ----
    A2 = np.array([[0.1, 1], [-1, 0.1]])
    s0 = np.array([0.02, 0.02])
    tspan = (0, 35)
    t_eval = np.arange(tspan[0], tspan[1] + tstep, tstep)
    sol2 = solve_ivp(
        lambda t, s: de_rhs(t, s, A2), tspan, s0, method="BDF", t_eval=t_eval
    )
    plt.figure(7)
    plt.plot(sol2.y[0, :], sol2.y[1, :], linewidth=2)
    plt.xlabel("u")
    plt.ylabel("v")
    # Plot nullclines as dashed lines
    plt.plot([-10, 10], [1, -1], "--", linewidth=2)
    plt.plot(
        [-1, 1], [-10, -10], "--", linewidth=2
    )  # note: slight adjustment for visual clarity
    plt.plot(0, 0, "*", markersize=12)
    plt.axis([-1, 1, -1, 1])
    plt.title("Figure 7: Unstable Spiral")

    # ---- Figure 8: Stable spiral ----
    A3 = np.array([[-0.1, 1], [-1, -0.1]])
    s0 = np.array([-0.4, -1])
    sol3 = solve_ivp(
        lambda t, s: de_rhs(t, s, A3), tspan, s0, method="BDF", t_eval=t_eval
    )
    plt.figure(8)
    plt.plot(sol3.y[0, :], sol3.y[1, :], linewidth=2)
    plt.xlabel("u")
    plt.ylabel("v")
    plt.plot([-10, 10], [-1, 1], "--", linewidth=2)
    plt.plot([-1, 1], [10, -10], "--", linewidth=2)
    plt.plot(0, 0, "*", markersize=12)
    plt.axis([-1, 1, -1, 1])
    plt.title("Figure 8: Stable Spiral")

    # ---- Figure 9: Stable node ----
    A4 = np.array([[-1, 0.2], [0.3, -0.3]])
    # Define initial data set (each column is an initial condition)
    idset = np.array([[0.4, -1, 1, -1], [-1, 0, 0, 1]])
    plt.figure(9)
    for j in range(idset.shape[1]):
        s0 = idset[:, j]
        sol4 = solve_ivp(
            lambda t, s: de_rhs(t, s, A4), tspan, s0, method="BDF", t_eval=t_eval
        )
        plt.plot(sol4.y[0, :], sol4.y[1, :], linewidth=2)
    plt.xlabel("u")
    plt.ylabel("v")
    # Add nullclines and markers
    plt.plot([-1, 1], [-5, 5], "--", linewidth=2)
    plt.plot([-10, 10], [-10, 10], "--", linewidth=2)
    plt.plot(0, 0, "*", markersize=12)
    plt.axis([-1, 1, -1, 1])
    # A couple of annotation arrows
    plt.gca().annotate(
        "",
        xy=(0.46, 0.27),
        xytext=(0.46, 0.19),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    plt.gca().annotate(
        "",
        xy=(0.57, 0.76),
        xytext=(0.57, 0.84),
        arrowprops=dict(arrowstyle="->", lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    plt.title("Figure 9: Stable Node Phase Portrait")

    plt.show()


# -------------------------------
# Run the plotting function if this script is executed
# -------------------------------
if __name__ == "__main__":
    make_de_plots()
