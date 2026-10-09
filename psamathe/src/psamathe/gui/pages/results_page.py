from psamathe.derived.orbit import compute_orbital_diagnostics
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from psamathe.derived.trajectory import (
    total_energy_history,
    relative_energy_error,
    linear_momentum_history,
    momentum_error,
    angular_momentum_history,
    angular_momentum_error,
    minimum_separation_history,
)


class ResultsPage(ttk.Frame):
    """Scientific summary and validation plots for a completed experiment."""

    def __init__(self, parent, app):
        super().__init__(parent)

        self.app = app

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # ---------------------------------------------------------
        # Header
        # ---------------------------------------------------------
        header = ttk.Frame(self)
        header.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 12),
        )

        ttk.Label(
            header,
            text="EXPERIMENT RESULTS",
            style="Title.TLabel",
        ).pack(side="left")

        ttk.Button(
            header,
            text="Run Another Experiment",
            command=self._return_to_configuration,
        ).pack(side="right")

        # ---------------------------------------------------------
        # Experiment summary
        # ---------------------------------------------------------
        self.summary = ttk.LabelFrame(
            self,
            text="Experiment Summary",
            padding=10,
        )

        self.summary.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(0, 10),
        )

        self.summary_label = ttk.Label(
            self.summary,
            text="No completed experiment.",
            justify="left",
            anchor="w",
        )

        self.summary_label.pack(
            fill="x",
        )

        # ---------------------------------------------------------
        # Scientific plots
        # ---------------------------------------------------------
        self.plot_frame = ttk.LabelFrame(
            self,
            text="Scientific Validation",
            padding=5,
        )

        self.plot_frame.grid(
            row=2,
            column=0,
            sticky="nsew",
        )

        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)

        self.figure = Figure(
            figsize=(10, 6),
            dpi=90,
            constrained_layout=True,
        )

        self.canvas = FigureCanvasTkAgg(
            self.figure,
            master=self.plot_frame,
        )

        self.canvas.get_tk_widget().grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        self.status_var = tk.StringVar(
            value="Waiting for a completed experiment."
        )

        ttk.Label(
            self,
            textvariable=self.status_var,
        ).grid(
            row=3,
            column=0,
            sticky="w",
            pady=(6, 0),
        )

    def on_show(self):
        """Refresh the summary and plots whenever this page is opened."""

        configuration = self.app.session.configuration
        result = self.app.session.result

        if configuration is None or result is None:
            self.summary_label.configure(
                text="No completed experiment is available."
            )

            self.status_var.set(
                "Run an experiment before viewing results."
            )

            self.figure.clear()
            self.canvas.draw_idle()
            return

        try:
            self._update_summary(configuration, result)
            self._update_plots(result)

            self.status_var.set(
                "Experiment results loaded."
            )

        except (ValueError, TypeError, IndexError) as error:
            self.status_var.set(
                f"Unable to display results: {error}"
            )

    def _update_summary(self, configuration, result):
        """Summarize the experiment and its numerical validation."""

        time = np.asarray(result.time, dtype=float)

        energies = np.asarray(
            total_energy_history(result),
            dtype=float,
        )

        energy_errors = np.asarray(
            relative_energy_error(result),
            dtype=float,
        )

        momentum_deviations = self._magnitude_history(
            momentum_error(result)
        )

        angular_deviations = self._magnitude_history(
            angular_momentum_error(result)
        )

        separations = np.asarray(
            minimum_separation_history(result),
            dtype=float,
        )

        initial_energy = energies[0]
        final_energy = energies[-1]

        max_energy_error = self._maximum_absolute(
            energy_errors
        )

        max_momentum_deviation = self._maximum_absolute(
            momentum_deviations
        )

        max_angular_deviation = self._maximum_absolute(
            angular_deviations
        )

        minimum_separation = self._minimum_finite(
            separations
        )

        n_steps = len(time) - 1
        actual_duration = time[-1] - time[0]

        summary_text = (
            f"Experiment: {configuration.name}\n"
            f"Status: Completed\n"
            f"Bodies: {result.n_bodies}\n"
            f"Integrator: {configuration.integrator}\n"
            f"Timestep: {configuration.dt:.6g} s\n"
            f"Requested duration: {configuration.duration:.6g} s\n"
            f"Actual duration: {actual_duration:.6g} s\n"
            f"Recorded integration steps: {n_steps}\n\n"
            f"Initial total energy: {initial_energy:.6e} J\n"
            f"Final total energy: {final_energy:.6e} J\n"
            f"Maximum absolute relative energy error: "
            f"{max_energy_error:.6e}\n"
            f"Maximum momentum deviation: "
            f"{max_momentum_deviation:.6e} kg m/s\n"
            f"Maximum angular momentum deviation: "
            f"{max_angular_deviation:.6e} kg m²/s\n"
            f"Minimum pairwise separation: "
            f"{minimum_separation:.6e} m"
        )

        if result.n_bodies == 2:
            orbital = compute_orbital_diagnostics(result)

            period_s = orbital["orbital_period_s"]

            if np.isfinite(period_s):
                period_days = period_s / 86400.0
                period_text = f"{period_days:.4f} days"
            else:
                period_text = "Unbound orbit"

            summary_text += (
                "\n\nOrbital Diagnostics (two-body model)\n"
                f"Bound orbit: {orbital['bound_orbit']}\n"
                f"Eccentricity: {orbital['eccentricity']:.8f}\n"
                f"Semi-major axis: "
                f"{orbital['semi_major_axis_m']:.6e} m\n"
                f"Estimated orbital period: {period_text}\n"
                f"Elapsed orbital periods: "
                f"{orbital['elapsed_periods']:.6f}\n"
                f"Maximum centre-of-mass drift: "
                f"{orbital['center_of_mass_drift_m']:.6e} m\n"
                f"Final relative-position difference: "
                f"{orbital['final_position_difference_m']:.6e} m\n"
                f"Final position difference / initial separation: "
                f"{orbital['final_position_difference_fraction']:.6e}"
            )
        self.summary_label.configure(
            text=summary_text
        )

    def _update_plots(self, result):
        """Build the six scientific plots from recorded simulation data."""

        time = np.asarray(
            result.time,
            dtype=float,
        )

        position = np.asarray(
            result.position,
            dtype=float,
        )

        energies = np.asarray(
            total_energy_history(result),
            dtype=float,
        )

        energy_errors = np.asarray(
            relative_energy_error(result),
            dtype=float,
        )

        momentum = np.asarray(
            linear_momentum_history(result),
            dtype=float,
        )

        momentum_deviations = self._magnitude_history(
            momentum_error(result)
        )

        angular_momentum = np.asarray(
            angular_momentum_history(result),
            dtype=float,
        )

        angular_deviations = self._magnitude_history(
            angular_momentum_error(result)
        )

        separations = np.asarray(
            minimum_separation_history(result),
            dtype=float,
        )

        self.figure.clear()

        axes = self.figure.subplots(
            2,
            3,
        )

        # ---------------------------------------------------------
        # 1. Two-dimensional trajectories
        # ---------------------------------------------------------
        ax = axes[0, 0]

        for body_index in range(result.n_bodies):
            ax.plot(
                position[:, body_index, 0],
                position[:, body_index, 1],
                label=f"Body {body_index + 1}",
                linewidth=1.2,
            )

            ax.scatter(
                position[0, body_index, 0],
                position[0, body_index, 1],
                marker="o",
                s=18,
            )

        ax.set_title("Trajectories")
        ax.set_xlabel("X position (m)")
        ax.set_ylabel("Y position (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)

        if result.n_bodies <= 8:
            ax.legend(fontsize=7)


        # ---------------------------------------------------------
        # 2. Total energy deviation from initial energy
        # ---------------------------------------------------------
        ax = axes[0, 1]

        energy_deviation = energies - energies[0]

        ax.plot(
            time,
            energy_deviation,
            linewidth=1.2,
        )

        ax.axhline(
            0.0,
            linewidth=0.8,
            linestyle="--",
        )

        ax.set_title("Total Energy Deviation")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ΔE (J)")
        ax.ticklabel_format(
            axis="y",
            style="sci",
            scilimits=(0, 0),
        )
        ax.grid(True, alpha=0.3)


        # ---------------------------------------------------------
        # 3. Relative energy error
        # ---------------------------------------------------------
        ax = axes[0, 2]

        ax.plot(
            time,
            energy_errors,
            linewidth=1.2,
        )

        ax.axhline(
            0.0,
            linewidth=0.8,
            linestyle="--",
        )

        ax.set_title("Relative Energy Error")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Relative error")
        ax.grid(True, alpha=0.3)


        # ---------------------------------------------------------
        # 4. Linear momentum deviation from initial momentum
        # ---------------------------------------------------------
        ax = axes[1, 0]

        momentum_deviation = np.asarray(
            momentum_error(result),
            dtype=float,
        )

        if (
            momentum_deviation.ndim == 2
            and momentum_deviation.shape[1] >= 2
        ):
            ax.plot(
                time,
                momentum_deviation[:, 0],
                label="ΔPx",
                linewidth=1.0,
            )

            ax.plot(
                time,
                momentum_deviation[:, 1],
                label="ΔPy",
                linewidth=1.0,
            )

            ax.legend(fontsize=8)

        else:
            ax.plot(
                time,
                momentum_deviation,
                label="Momentum deviation",
                linewidth=1.0,
            )

        ax.axhline(
            0.0,
            linewidth=0.8,
            linestyle="--",
        )

        ax.set_title("Linear Momentum Deviation")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Deviation (kg m/s)")
        ax.ticklabel_format(
            axis="y",
            style="sci",
            scilimits=(0, 0),
        )
        ax.grid(True, alpha=0.3)

        # ---------------------------------------------------------
        # 5. Angular momentum deviation from initial value
        # ---------------------------------------------------------
        ax = axes[1, 1]

        angular_deviation = np.asarray(
            angular_momentum_error(result),
            dtype=float,
        )

        ax.plot(
            time,
            angular_deviation,
            linewidth=1.2,
            label="ΔLz",
        )

        ax.axhline(
            0.0,
            linewidth=0.8,
            linestyle="--",
        )

        ax.set_title("Angular Momentum Deviation")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Deviation (kg m²/s)")
        ax.ticklabel_format(
            axis="y",
            style="sci",
            scilimits=(0, 0),
        )
        ax.grid(True, alpha=0.3)


        # ---------------------------------------------------------
        # 6. Minimum pairwise separation
        # ---------------------------------------------------------
        ax = axes[1, 2]

        ax.plot(
            time,
            separations,
            linewidth=1.2,
        )

        ax.set_title("Minimum Separation")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance (m)")
        ax.grid(True, alpha=0.3)

        self.canvas.draw_idle()

    @staticmethod
    def _magnitude_history(values):
        """
        Convert vector deviations to magnitudes.

        Scalar histories remain unchanged.
        """

        values = np.asarray(
            values,
            dtype=float,
        )

        if values.ndim > 1:
            return np.linalg.norm(
                values.reshape(values.shape[0], -1),
                axis=1,
            )

        return values

    @staticmethod
    def _maximum_absolute(values):
        """Return the maximum finite absolute value."""

        values = np.asarray(
            values,
            dtype=float,
        )

        finite_values = values[
            np.isfinite(values)
        ]

        if finite_values.size == 0:
            return float("nan")

        return float(
            np.max(np.abs(finite_values))
        )

    @staticmethod
    def _minimum_finite(values):
        """Return the minimum finite value."""

        values = np.asarray(
            values,
            dtype=float,
        )

        finite_values = values[
            np.isfinite(values)
        ]

        if finite_values.size == 0:
            return float("nan")

        return float(
            np.min(finite_values)
        )

    def _return_to_configuration(self):
        """Return to the configuration page."""

        self.app.show_page(
            "ConfigPage"
        )
