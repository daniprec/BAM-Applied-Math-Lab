import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.widgets import Slider
from PIL import Image


def laplacian(uv: np.ndarray) -> np.ndarray:
    """
    Compute the Laplacian of the u and v fields using a 5-point finite
    difference scheme: considering each point and its immediate neighbors in
    the up, down, left, and right directions.

    Reference: https://en.wikipedia.org/wiki/Five-point_stencil

    Parameters
    ----------
    uv : np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the values of
        u and v.
    Returns
    -------
    np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the Laplacian
        of u and v.
    """
    lap = -4 * uv

    # Immediate neighbors (up, down, left, right)
    lap += np.roll(uv, shift=1, axis=1)  # up
    lap += np.roll(uv, shift=-1, axis=1)  # down
    lap += np.roll(uv, shift=1, axis=2)  # left
    lap += np.roll(uv, shift=-1, axis=2)  # right
    return lap


def laplacian_9pt(uv: np.ndarray) -> np.ndarray:
    """
    Compute the Laplacian of the u and v fields using a 9-point finite
    difference scheme (Patra-Karttunen), considering each point and its
    immediate neighbors, including diagonals.

    Reference: https://en.wikipedia.org/wiki/Nine-point_stencil

    Parameters
    ----------
    uv : np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the values of u and v.

    Returns
    -------
    np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the Laplacian of u and v.
    """
    # Weights for the 9-point stencil (Patra-Karttunen)
    center_weight = -20 / 6
    neighbor_weight = 4 / 6
    diagonal_weight = 1 / 6

    lap = center_weight * uv

    # Shifted arrays for immediate neighbors
    up = np.roll(uv, shift=1, axis=1)
    down = np.roll(uv, shift=-1, axis=1)

    # Immediate neighbors (up, down, left, right)
    lap += neighbor_weight * up  # up
    lap += neighbor_weight * down  # down
    lap += neighbor_weight * np.roll(uv, shift=1, axis=2)  # left
    lap += neighbor_weight * np.roll(uv, shift=-1, axis=2)  # right

    # Diagonal neighbors
    lap += diagonal_weight * np.roll(up, shift=1, axis=2)  # up-left
    lap += diagonal_weight * np.roll(up, shift=-1, axis=2)  # up-right
    lap += diagonal_weight * np.roll(down, shift=1, axis=2)  # down-left
    lap += diagonal_weight * np.roll(down, shift=-1, axis=2)  # down-right

    return lap


def gray_scott_pde(
    t: float,
    uv: np.ndarray,
    mask: np.ndarray,
    d1: float = 0.1,
    d2: float = 0.05,
    f: tuple[float, float] = (0.040, 0.060),
    k: tuple[float, float] = (0.060, 0.040),
    stencil: int = 5,
) -> np.ndarray:
    """
    Update the u and v fields using the Gray-Scott model with explicit Euler
    time integration, where the fields are updated based on their current values
    and the calculated derivatives.

    Reference: https://itp.uni-frankfurt.de/~gros/StudentProjects/Projects_2020/projekt_schulz_kaefer/

    Parameters
    ----------
    uv : np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the values of
        u and v.
    mask : np.ndarray
        2D array with shape (grid_size, grid_size) containing the mask for the
        u and v fields. Each element is either 0 or 1, indicating the parameter
        to use from the tuples f and k.
    d1 : float, optional
        Diffusion rate of u, default is 0.1.
    d2 : float, optional
        Diffusion rate of v, default is 0.05.
    f : tuple[float,float], optional
        Feed rate (at which u is fed into the system), default is 0.040.
    k : tuple[float,float], optional
        Kill rate (at which v is removed from the system), default is 0.060.
    stencil : int, optional
        Stencil to use for the Laplacian computation. Use 5 or 9, default is 5.
    """

    # Extract the matrices for substances u and v
    u, v = uv

    # Generate the matrices for f and k based on the mask
    f_matrix = np.ones_like(u) * f[0]
    f_matrix[mask == 1] = f[1]
    k_matrix = np.ones_like(v) * k[0]
    k_matrix[mask == 1] = k[1]

    # Compute the Laplacian of u and v
    if stencil == 5:
        lap = laplacian(uv)
    elif stencil == 9:
        lap = laplacian_9pt(uv)
    else:
        raise ValueError("Invalid stencil value. Use 5 or 9.")

    # Extract the Laplacian matrices for u and v
    lu, lv = lap

    uv2 = u * v * v

    # Gray-Scott equations
    du_dt = d1 * lu - uv2 + f_matrix * (1 - u)
    dv_dt = d2 * lv + uv2 - (f_matrix + k_matrix) * v

    return np.array([du_dt, dv_dt])


def run_simulation(
    n: int = 250,
    dx: float = 1,
    dt: float = 2,
    anim_speed: int = 100,
    cmap: str = "jet",
):
    """
    Animate the Gray-Scott model simulation.

    Parameters
    ----------
    n : int
        Number of grid points in one dimension, N.
    dx : float
        Spacing between grid points.
    dt : float
        Time step.
    boundary_conditions : str
        Boundary conditions to apply. Use 'neumann' or 'periodic'.
    anim_speed : int
        Animation speed. Number of iterations per frame.
    cmap : str
        Colormap to use for the plot, by default 'jet'.
    """
    # We first ask the user to upload an image
    # that will be used as a mask for the Gray-Scott model simulation.

    # Open a file dialog to select an image
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select an image for the mask",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
            ("All files", "*.*"),  # Fallback
        ],
    )
    # The following avoids errors if the user cancels the file picker
    if not file_path:
        print("No file selected. Exiting.")
        return

    root.destroy()

    # Load and process the image
    img = Image.open(file_path).convert("L")  # Convert to grayscale
    img = img.resize((n, n), Image.Resampling.LANCZOS)
    img_np = np.array(img)
    # Threshold to create a binary mask (0 or 1)
    threshold = 128
    mask = (img_np > threshold).astype(np.int8)

    # ------------------------------------------------------------------------#
    # PARAMETERS
    # ------------------------------------------------------------------------#
    length = n * dx  # L

    # Initial parameters - Will be changed using the sliders
    d1 = 0.1
    d2 = 0.05
    f = (0.040, 0.060)
    k = (0.060, 0.040)

    # Pause parameter - Will be toggled by pressing the space bar (see on_keypress)
    pause = False

    # Drawing parameter - Will be toggled by clicking on the plot (see on_click, on_release)
    drawing = False

    # ------------------------------------------------------------------------#
    # INITIALIZE THE PLOT
    # ------------------------------------------------------------------------#

    # Initialize the (u, v) = (1, 0)
    uv = np.ones((2, length, length), dtype=np.float32)
    uv[1] = 0

    # Create figure with plot on the left (6x6) and sliders on the right (6x4)
    fig, axs = plt.subplots(
        nrows=1, ncols=2, figsize=(10, 6), gridspec_kw={"width_ratios": [6, 4]}
    )

    # Initialize the v field image
    ax_uv: Axes = axs[0]
    ax_uv.axis("off")  # Turn off the axis (the grid and numbers)
    im = ax_uv.imshow(uv[1], cmap=cmap, interpolation="bilinear", vmin=0, vmax=1.0)

    # ------------------------------------------------------------------------#
    # ANIMATION
    # ------------------------------------------------------------------------#

    def update_frame(_):
        """This function is called by the animation module at each frame.
        It updates the u and v fields based on the Gray-Scott model equations.
        Returns the updated elements of the plot. This in combination with the
        blit=True parameter in FuncAnimation will only update the changed elements
        of the plot, making the animation faster.
        """
        # Access variables from the outer scope
        nonlocal pause, uv, mask, d1, d2, f, k
        if pause:
            return [im]

        for _ in range(anim_speed):
            # Solve an initial value problem for a system of ODEs via Euler's method
            uv = uv + gray_scott_pde(_, uv, mask=mask, d1=d1, d2=d2, f=f, k=k) * dt
            # Neumann - zero flux boundary conditions
            uv[:, 0, :] = uv[:, 1, :]
            uv[:, -1, :] = uv[:, -2, :]
            uv[:, :, 0] = uv[:, :, 1]
            uv[:, :, -1] = uv[:, :, -2]

        im.set_array(uv[1])
        return [im]  # Elements to update (using blit=True)

    # Create the animation
    ani = animation.FuncAnimation(fig, update_frame, interval=1, blit=True)

    # ------------------------------------------------------------------------#
    # ANIMATION - PAUSE / RESUME
    # ------------------------------------------------------------------------#

    # We want the user to be able to pause or resume the simulation by pressing the space bar
    # In order to do this, we need a key press event handler: on_keypress

    def on_keypress(event: KeyEvent):
        """This function is called when the user presses a key.
        It pauses or resumes the simulation when the space bar is pressed."""
        # Pressing the space bar pauses or resumes the simulation
        if event.key == " ":
            nonlocal pause
            pause ^= True

    # Attach the key press event handler to the figure
    fig.canvas.mpl_connect("key_press_event", on_keypress)

    # ------------------------------------------------------------------------#
    # SLIDERS
    # ------------------------------------------------------------------------#

    # The following sliders will allow the user to change the parameters of the Gray-Scott model

    # Create the sliders axes
    ax_sliders: Axes = axs[1]
    ax_sliders.axis("off")  # Turn off the axis (the grid and numbers)

    # Place the axes objects that will contain the sliders
    # We define the location of each axes inside the right column of the figure
    ax_d1 = ax_sliders.inset_axes([0.0, 0.85, 0.8, 0.1])  # [x0, y0, width, height]
    ax_d2 = ax_sliders.inset_axes([0.0, 0.75, 0.8, 0.1])  # [x0, y0, width, height]
    ax_f0 = ax_sliders.inset_axes([0.0, 0.6, 0.8, 0.1])  # [x0, y0, width, height]
    ax_f1 = ax_sliders.inset_axes([0.0, 0.5, 0.8, 0.1])  # [x0, y0, width, height]
    ax_k0 = ax_sliders.inset_axes([0.0, 0.35, 0.8, 0.1])  # [x0, y0, width, height]
    ax_k1 = ax_sliders.inset_axes([0.0, 0.25, 0.8, 0.1])  # [x0, y0, width, height]
    ax_thresh = ax_sliders.inset_axes([0.2, 0.1, 0.6, 0.1])  # [x0, y0, width, height]

    # Create the sliders, each in its own axes [min, max, initial]
    slider_f0 = Slider(ax_f0, "F0", 0.01, 0.09, valinit=f[0], valstep=0.01)
    slider_f1 = Slider(ax_f1, "F1", 0.01, 0.09, valinit=f[1], valstep=0.01)
    slider_k0 = Slider(ax_k0, "K0", 0.01, 0.07, valinit=k[0], valstep=0.01)
    slider_k1 = Slider(ax_k1, "K1", 0.01, 0.07, valinit=k[1], valstep=0.01)
    slider_d1 = Slider(ax_d1, "D1", 0.01, 0.2, valinit=d1, valstep=0.01)
    slider_d2 = Slider(ax_d2, "D2", 0.01, 0.2, valinit=d2, valstep=0.01)
    slider_thresh = Slider(ax_thresh, "Brightness", 0, 100, valinit=50, valstep=1)

    # This function will be called when the user changes the sliders
    def update_sliders(_):
        # Acces the variables from the outer scope to update them
        nonlocal f, k, d1, d2, threshold, mask, pause
        # Update the parameters according to the sliders values
        f = (slider_f0.val, slider_f1.val)
        k = (slider_k0.val, slider_k1.val)
        d1 = slider_d1.val
        d2 = slider_d2.val
        threshold = 255 - 255 * slider_thresh.val / 100
        # Update the mask
        # Threshold to create a binary mask (0 or 1)
        mask = (img_np > threshold).astype(np.int8)
        # Pause the simulation when sliders are updated
        # pause = True

    # Attach the update function to sliders
    slider_f0.on_changed(update_sliders)
    slider_f1.on_changed(update_sliders)
    slider_k0.on_changed(update_sliders)
    slider_k1.on_changed(update_sliders)
    slider_d1.on_changed(update_sliders)
    slider_d2.on_changed(update_sliders)
    slider_thresh.on_changed(update_sliders)

    # ------------------------------------------------------------------------#
    #  DRAW ON UV PLOT
    # ------------------------------------------------------------------------#

    # We want the user to be able to add and remove sources of v by clicking on the plot
    # The first function we define is update_uv, which will be called when the user clicks on the plot
    # It will update the u and v fields based on the mouse position and the mouse button pressed
    # Right click will remove the source of v, left click will add a source of v

    def update_uv(event: MouseEvent, r: int = 3):
        """Update the u and v fields based on the mouse position.

        Parameters
        ----------
        event
            The event object.
        r : int
            Radius of the source, by default 3.
        """
        if event.inaxes != ax_uv:
            return
        if event.xdata is None or event.ydata is None:
            return
        x = int(event.xdata)
        y = int(event.ydata)

        # Left click? Add a perturbation (u, v) = (0.5, 0.5)
        # plus an additive noise of 10% that value
        if event.button == 1:
            u_new = 0.5 * (1 + 0.1 * np.random.randn())
            v_new = 0.5 * (1 + 0.1 * np.random.randn())
        # Right click? Reset to initial values (u, v) = (1, 0)
        elif event.button == 3:
            u_new = 1.0
            v_new = 0.0
        else:
            return

        uv[0, y - r : y + r, x - r : x + r] = u_new
        uv[1, y - r : y + r, x - r : x + r] = v_new

        # Update the displayed image
        im.set_array(uv[1])

    # Next, we want the user to be able to draw lines on the plot to add or remove sources of v
    # In order to do this, we need to define some mouse event handlers: on_click, on_release, on_motion

    def on_click(event: MouseEvent):
        """This function is called when the user clicks on the plot.
        It initializes the drawing process."""
        if event.inaxes != ax_uv:
            return
        if event.xdata is None or event.ydata is None:
            return
        nonlocal drawing
        drawing = True
        update_uv(event)

    def on_release(event: MouseEvent):
        """This function is called when the user releases the mouse button.
        It stops the drawing process."""
        nonlocal drawing
        drawing = False

    def on_motion(event: MouseEvent):
        """This function is called when the user moves the mouse.
        It updates the u and v fields if the drawing process is active.
        """
        if drawing:
            update_uv(event)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)

    # ------------------------------------------------------------------------#
    #  DISPLAY
    # ------------------------------------------------------------------------#

    plt.show()


if __name__ == "__main__":
    run_simulation()
