"""IE University colours and a small Matplotlib theme for the course animations.

Palette (from the IE brand sheet):

    IE ocean blue   #000066     IE electric blue  #0000DB     IE sea blue  #47BFFF
    Tech green      #6DC201     Tech green (new)  #1F3700     Tech black   #000024
    VIG red         #E2000F

Two modes are provided. ``"dark"`` uses Tech black as the stage, which makes
moving agents easy to follow in a lecture room. ``"light"`` uses white and
ocean blue, which is better for printing and for slides with a white background.

Accessibility notes (WCAG 2.1 AA, as required by the IE design system):

* Meaning is never carried by colour alone. The predator is a larger shape with
  an outline, the odd prey has a ring, and every plot has a text readout.
* Sea blue is used for graphics and for text *on dark backgrounds only*, never
  as text on white.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# --- Brand colours ---------------------------------------------------------
OCEAN_BLUE = "#000066"
ELECTRIC_BLUE = "#0000DB"
SEA_BLUE = "#47BFFF"
TECH_GREEN = "#6DC201"
TECH_GREEN_NEW = "#1F3700"
TECH_BLACK = "#000024"
VIG_RED = "#E2000F"

# Greys from the IE design system
IE_BLACK = "#0F0F0F"
DARK_GREY = "#434445"
MIDDLE_GREY = "#888A8C"
LIGHT_GREY = "#BFC1C3"
IE_WHITE = "#F4F4F4"
WHITE = "#FFFFFF"

# A lighter red that still reads as "VIG red" on the Tech-black stage.
VIG_RED_GLOW = "#FF5A63"


def palette(mode: str = "dark") -> dict[str, str]:
    """Colour roles used by the animations.

    Parameters
    ----------
    mode : {"dark", "light"}
    """
    if mode == "dark":
        return {
            "bg": TECH_BLACK,
            "panel": "#05053A",  # one step above the stage
            "floor": "#08083F",
            "text": WHITE,
            "muted": "#A9B4D6",
            "grid": "#1B1B5C",
            "prey": SEA_BLUE,
            "prey_dim": "#2C7AA8",
            "odd": TECH_GREEN,
            "predator": VIG_RED_GLOW,
            "predator_edge": WHITE,
            "accent": SEA_BLUE,
            "good": TECH_GREEN,
            "bad": VIG_RED_GLOW,
            "wall": SEA_BLUE,
            "slider": SEA_BLUE,
            "slider_bg": "#1B1B5C",
        }
    if mode == "light":
        return {
            "bg": WHITE,
            "panel": IE_WHITE,
            "floor": "#EAF3FB",
            "text": IE_BLACK,
            "muted": DARK_GREY,
            "grid": LIGHT_GREY,
            "prey": ELECTRIC_BLUE,
            "prey_dim": "#6F6FD8",
            "odd": "#2F6A00",
            "predator": VIG_RED,
            "predator_edge": IE_BLACK,
            "accent": OCEAN_BLUE,
            "good": "#2F6A00",
            "bad": VIG_RED,
            "wall": OCEAN_BLUE,
            "slider": ELECTRIC_BLUE,
            "slider_bg": LIGHT_GREY,
        }
    raise ValueError(f"mode must be 'dark' or 'light', got {mode!r}")


def pressure_cmap(mode: str = "dark") -> LinearSegmentedColormap:
    """Low pressure = calm blue, high pressure = VIG red (via a bright midpoint)."""
    if mode == "dark":
        colors = [SEA_BLUE, "#E8F6FF", VIG_RED_GLOW]
    else:
        colors = [ELECTRIC_BLUE, "#B9A6FF", VIG_RED]
    return LinearSegmentedColormap.from_list("ie_pressure", colors)


def apply_theme(mode: str = "dark") -> dict[str, str]:
    """Set Matplotlib rcParams for the chosen mode and return its palette."""
    p = palette(mode)
    mpl.rcParams.update(
        {
            "figure.facecolor": p["bg"],
            "axes.facecolor": p["panel"],
            "axes.edgecolor": p["grid"],
            "axes.labelcolor": p["text"],
            "axes.titlecolor": p["text"],
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": p["grid"],
            "grid.linewidth": 0.6,
            "xtick.color": p["muted"],
            "ytick.color": p["muted"],
            "text.color": p["text"],
            "font.family": "sans-serif",
            # Montserrat is the IE UI font; fall back gracefully when absent.
            "font.sans-serif": ["Montserrat", "Arial", "Helvetica", "DejaVu Sans"],
            "legend.facecolor": p["panel"],
            "legend.edgecolor": p["grid"],
            "legend.labelcolor": p["text"],
            "savefig.facecolor": p["bg"],
        }
    )
    return p


def style_slider(slider, p: dict[str, str]) -> None:
    """Colour a Matplotlib Slider with the IE palette."""
    slider.poly.set_facecolor(p["slider"])
    slider.track.set_facecolor(p["slider_bg"])
    slider.label.set_color(p["text"])
    slider.valtext.set_color(p["text"])
    slider.label.set_fontsize(9)
    slider.valtext.set_fontsize(9)
    handle = getattr(slider, "_handle", None)
    if handle is not None:
        handle.set_markerfacecolor(p["slider"])
        handle.set_markeredgecolor(p["text"])
