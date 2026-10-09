import tkinter as tk
from tkinter import ttk

import numpy as np

from psamathe.derived.physical import pairwise_distances
from psamathe.simulation.factory import create_simulation_engine
from psamathe.simulation.recorder import SimulationRecorder


class LivePage(ttk.Frame):
    """
    Live visualization and execution page for a simulation experiment.

    The page controls execution timing and visualization, but does not
    perform any physics calculations itself.
    """

    UPDATE_INTERVAL_MS = 20

    def __init__(self, parent, controller):
        super().__init__(parent)

        self.controller = controller

        self.engine = None
        self.recorder = None

        self.running = False
        self.finished = False
        self.after_id = None
        
        self.canvas_width = 700
        self.canvas_height = 600
        self.trail_lines = {}

        self.canvas = tk.Canvas(
            self,
            width=self.canvas_width,
            height=self.canvas_height,
            background="black",
            highlightthickness=0,
        )

        self.canvas.grid(
            row=0,
            column=0,
            rowspan=2,
            sticky="nsew",
            padx=(10, 5),
            pady=10,
        )

        self._build_control_panel()

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def _build_control_panel(self):
        """Build the runtime information and controls."""

        panel = ttk.Frame(self)
        panel.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(5, 10),
            pady=10,
        )

        ttk.Label(
            panel,
            text="LIVE SIMULATION",
            font=("TkDefaultFont", 14, "bold"),
        ).pack(anchor="w", pady=(0, 20))

        self.time_var = tk.StringVar(value="0.00 s")
        self.dt_var = tk.StringVar(value="-")
        self.integrator_var = tk.StringVar(value="-")
        self.body_count_var = tk.StringVar(value="-")
        self.min_separation_var = tk.StringVar(value="-")
        self.status_var = tk.StringVar(value="Ready")

        self._add_telemetry_row(
            panel,
            "Simulation Time",
            self.time_var,
        )

        self._add_telemetry_row(
            panel,
            "Timestep",
            self.dt_var,
        )

        self._add_telemetry_row(
            panel,
            "Integrator",
            self.integrator_var,
        )

        self._add_telemetry_row(
            panel,
            "Bodies",
            self.body_count_var,
        )

        self._add_telemetry_row(
            panel,
            "Minimum Separation",
            self.min_separation_var,
        )

        ttk.Separator(panel).pack(
            fill="x",
            pady=15,
        )

        ttk.Label(
            panel,
            text="Status",
        ).pack(anchor="w")

        ttk.Label(
            panel,
            textvariable=self.status_var,
        ).pack(anchor="w", pady=(2, 15))

        self.pause_button = ttk.Button(
            panel,
            text="Pause",
            command=self.toggle_pause,
        )

        self.pause_button.pack(
            fill="x",
            pady=3,
        )

        ttk.Button(
            panel,
            text="Stop",
            command=self.stop_simulation,
        ).pack(
            fill="x",
            pady=3,
        )

        ttk.Button(
            panel,
            text="Return to Configuration",
            command=self.return_to_configuration,
        ).pack(
            fill="x",
            pady=(20, 3),
        )

    def _add_telemetry_row(self, parent, label, variable):
        """Add one runtime telemetry field."""

        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=4)

        ttk.Label(
            frame,
            text=label,
        ).pack(anchor="w")

        ttk.Label(
            frame,
            textvariable=variable,
        ).pack(anchor="w")

    def on_show(self):
        """
        Called whenever the page becomes visible.
        """

        configuration = self.controller.session.configuration

        if configuration is None:
            self.controller.show_page("ConfigPage")
            return

        self._start_experiment(configuration)

    def _start_experiment(self, configuration):
        """Create the engine and begin incremental execution."""
        self.after_id = None
        self.trail_lines = {}

        if hasattr(self, "display_scale"):
            del self.display_scale

            
        self.engine = create_simulation_engine(configuration)

        self.recorder = SimulationRecorder(
            self.engine.state
        )

        self.running = True
        self.finished = False

        self.pause_button.configure(
            text="Pause"
        )

        self.status_var.set("Running")

        self.time_var.set(
            f"{self.engine.state.time:.2f} s"
        )

        self.dt_var.set(
            f"{self.engine.dt:.6g} s"
        )

        self.integrator_var.set(
            configuration.integrator
        )

        self.body_count_var.set(
            str(configuration.n_bodies)
        )

        self._reset_canvas()

        self._draw_state()

        self._schedule_update()

    def _schedule_update(self):
        """Schedule the next simulation update."""

        if self.running and not self.finished:
            self.after_id = self.after(
                self.UPDATE_INTERVAL_MS,
                self._update_simulation,
            )
    
    def _update_simulation(self):
        """Advance the simulation and refresh the visualization."""

        self.after_id = None

        if not self.running:
            return

        if self.finished:
            return

        configuration = self.controller.session.configuration

        if self.engine.state.time >= configuration.duration:
            self._finish_simulation()
            return

        self.engine.step()

        self.recorder.record(
            self.engine.state
        )

        self._draw_state()

        self.time_var.set(
            f"{self.engine.state.time:.2f} s"
        )

        if self.engine.state.time >= configuration.duration:
            self._finish_simulation()
            return

        self._schedule_update()

    def _draw_state(self):
        """Render the current simulation state."""

        state = self.engine.state
        positions = state.position

        self._update_scale(positions)

        self.canvas.delete("bodies")

        for index, position in enumerate(positions):
            x, y = self._world_to_canvas(
                position[0],
                position[1],
        )

            radius = 6

            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="white",
                outline="",
                tags="bodies",
            )

            self.canvas.create_text(
                x + 10,
                y - 10,
                text=str(index + 1),
                fill="white",
                anchor="w",
                tags="bodies",
            )

        self._update_trails()

        minimum_separation = self._minimum_separation(
            positions
        )

        if np.isfinite(minimum_separation):
            self.min_separation_var.set(
                f"{minimum_separation:.6e} m"
            )
        else:
            self.min_separation_var.set(
                "N/A"
            )

    def _update_trails(self):
        """
    Update trajectory trails incrementally.

    Only the newest point is added to each trail rather than
    rebuilding the complete trajectory every frame.
    """

        if self.recorder is None:
            return

        positions = self.recorder.position_history

        if len(positions) == 0:
            return

        current_positions = positions[-1]

        for body_index, position in enumerate(current_positions):
            x, y = self._world_to_canvas(
                position[0],
                position[1],
            )

            if body_index not in self.trail_lines:
                line_id = self.canvas.create_line(
                    x,
                    y,
                    x,
                    y,
                    fill="gray",
                    width=1,
                    tags="trails",
                )

                self.trail_lines[body_index] = [
                    line_id,
                    [x, y],
                ]

            else:
                line_id, points = self.trail_lines[body_index]

                points.extend([x, y])

                self.canvas.coords(
                    line_id,
                    *points,
                )

    def _update_scale(self, positions):
        """
    Determine the display scale once for the experiment.

    The scale is based on the initial configuration so that
    trajectories remain spatially consistent throughout the run.
    """

        if hasattr(self, "display_scale"):
            return

        maximum_distance = np.max(
            np.abs(positions)
        )

        if maximum_distance <= 0:
            self.display_scale = 1.0
        else:
            self.display_scale = (
                min(
                    self.canvas_width,
                    self.canvas_height,
                )
                * 0.4
                / maximum_distance
            )

    def _world_to_canvas(self, x, y):
        """Convert SI coordinates into Canvas coordinates."""

        center_x = self.canvas_width / 2
        center_y = self.canvas_height / 2

        scale = getattr(
            self,
            "display_scale",
            1.0,
        )

        canvas_x = center_x + x * scale
        canvas_y = center_y - y * scale

        return canvas_x, canvas_y

    @staticmethod
    def _minimum_separation(positions):
        """Return the minimum pairwise body separation."""

        if len(positions) < 2:
            return np.nan

        displacement = (
            positions[:, np.newaxis, :]
            - positions[np.newaxis, :, :]
        )

        distances = np.linalg.norm(
            displacement,
            axis=2,
        )

        distances[
            np.diag_indices_from(distances)
        ] = np.inf

        return np.min(distances)

    def toggle_pause(self):
        """Pause or resume the simulation."""

        if self.finished:
            return

        if self.running:
            self.running = False

            if self.after_id is not None:
                self.after_cancel(
                    self.after_id
                )
                self.after_id = None

            self.pause_button.configure(
                text="Resume"
            )

            self.status_var.set(
                "Paused"
            )

        else:
            self.running = True

            self.pause_button.configure(
                text="Pause"
            )

            self.status_var.set(
                "Running"
            )

            self._schedule_update()

    def stop_simulation(self):
        """Stop the current simulation."""

        self.running = False
        self.finished = True

        if self.after_id is not None:
            self.after_cancel(
                self.after_id
            )
            self.after_id = None

        self.pause_button.configure(
            text="Pause"
        )

        self.status_var.set(
            "Stopped"
        )

    def _finish_simulation(self):
        """Finalize the experiment."""

        self.running = False
        self.finished = True

        self.status_var.set(
            "Completed"
        )

        self.pause_button.configure(
            text="Pause"
        )

        self.controller.session.result = (
            self.recorder.result()
        )

        self.controller.show_page(
            "ResultsPage"
        )

    def _reset_canvas(self):
        """Clear the visualization and reset display state."""

        self.canvas.delete("all")

        self.trail_lines = {}

        if hasattr(self, "display_scale"):
            del self.display_scale

    def return_to_configuration(self):
        """Return to the configuration page."""

        self.running = False
        self.finished = True

        if self.after_id is not None:
            self.after_cancel(
                self.after_id
            )
            self.after_id = None

        self.controller.show_page(
            "ConfigPage"
        )