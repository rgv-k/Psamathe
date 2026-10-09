import tkinter as tk
from tkinter import messagebox, ttk

from psamathe.gui.state import ExperimentConfiguration


class ConfigPage(ttk.Frame):
    """Experiment configuration page."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        self.name_var = tk.StringVar(value="Two Body Validation")
        self.scenario_var = tk.StringVar(value="Custom")
        self.n_bodies_var = tk.IntVar(value=2)
        self.duration_var = tk.StringVar(value="1000")
        self.integrator_var = tk.StringVar(value="Velocity Verlet")
        self.dt_var = tk.StringVar(value="1.0")

        self.body_entries = []

        self._build()
        self._rebuild_body_table()

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 18))
        header.columnconfigure(0, weight=1)

        ttk.Label(
            header,
            text="PSAMATHE",
            style="Title.TLabel",
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(
            header,
            text="PHASE 01  /  EXPERIMENT CONFIGURATION",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        experiment = ttk.LabelFrame(self, text="Experiment")
        experiment.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        experiment.columnconfigure(1, weight=1)

        ttk.Label(experiment, text="Name").grid(
            row=0, column=0, sticky="w", padx=10, pady=8
        )
        ttk.Entry(experiment, textvariable=self.name_var).grid(
            row=0, column=1, sticky="ew", padx=10, pady=8
        )

        ttk.Label(experiment, text="Scenario").grid(
            row=1, column=0, sticky="w", padx=10, pady=8
        )
        
        self.scenario_combo = ttk.Combobox(
            experiment,
            textvariable=self.scenario_var,
            values=("Custom", "Earth-Sun (perihelion)", "Three-Body Figure Eight"),
            state="readonly",
        )

        self.scenario_combo.grid(
            row=1,
            column=1,
            sticky="w",
            padx=10,
            pady=8,
        )

        self.scenario_combo.bind(
            "<<ComboboxSelected>>",
            self._on_scenario_selected,
        )


        numerical = ttk.LabelFrame(self, text="Numerical Configuration")
        numerical.grid(row=2, column=0, sticky="ew", pady=(0, 12))

        ttk.Label(numerical, text="Bodies").grid(
            row=0, column=0, sticky="w", padx=10, pady=8
        )

        bodies_spin = ttk.Spinbox(
            numerical,
            from_=2,
            to=100,
            textvariable=self.n_bodies_var,
            width=10,
            command=self._rebuild_body_table,
        )
        bodies_spin.grid(row=0, column=1, sticky="w", padx=10, pady=8)
        bodies_spin.bind("<Return>", lambda _event: self._rebuild_body_table())
        bodies_spin.bind("<FocusOut>", lambda _event: self._rebuild_body_table())

        ttk.Label(numerical, text="Duration (s)").grid(
            row=0, column=2, sticky="w", padx=(30, 10), pady=8
        )
        ttk.Entry(numerical, textvariable=self.duration_var, width=14).grid(
            row=0, column=3, sticky="w", padx=10, pady=8
        )

        ttk.Label(numerical, text="Integrator").grid(
            row=1, column=0, sticky="w", padx=10, pady=8
        )
        ttk.Combobox(
            numerical,
            textvariable=self.integrator_var,
            values=("Euler", "Velocity Verlet", "RK4"),
            state="readonly",
            width=18,
        ).grid(row=1, column=1, sticky="w", padx=10, pady=8)

        ttk.Label(numerical, text="Timestep (s)").grid(
            row=1, column=2, sticky="w", padx=(30, 10), pady=8
        )
        ttk.Entry(numerical, textvariable=self.dt_var, width=14).grid(
            row=1, column=3, sticky="w", padx=10, pady=8
        )

        ttk.Label(numerical, text="Precision: float64").grid(
            row=2, column=0, columnspan=2, sticky="w", padx=10, pady=8
        )
        ttk.Label(numerical, text="Units: SI").grid(
            row=2, column=2, columnspan=2, sticky="w", padx=(30, 10), pady=8
        )

        self.body_frame = ttk.LabelFrame(self, text="Initial Body Conditions")
        self.body_frame.grid(row=4, column=0, sticky="nsew", pady=(0, 12))
        self.body_frame.columnconfigure(0, weight=1)
        self.body_frame.rowconfigure(0, weight=1)

        footer = ttk.Frame(self)
        footer.grid(row=5, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="READY")
        ttk.Label(
            footer,
            textvariable=self.status_var,
            style="Telemetry.TLabel",
        ).grid(row=0, column=0, sticky="w")

        ttk.Button(
            footer,
            text="START EXPERIMENT",
            style="Primary.TButton",
            command=self._start_experiment,
        ).grid(row=0, column=1, sticky="e")

    def _rebuild_body_table(self):
        try:
            n = int(self.n_bodies_var.get())
        except (ValueError, tk.TclError):
            return

        if n < 2:
            return

        for widget in self.body_frame.winfo_children():
            widget.destroy()

        canvas = tk.Canvas(self.body_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            self.body_frame,
            orient="vertical",
            command=canvas.yview,
        )
        table = ttk.Frame(canvas)

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.body_frame.columnconfigure(0, weight=1)
        self.body_frame.rowconfigure(0, weight=1)

        canvas_window = canvas.create_window((0, 0), window=table, anchor="nw")

        table.bind(
            "<Configure>",
            lambda _event: canvas.configure(
                scrollregion=canvas.bbox("all")
            ),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(
                canvas_window,
                width=event.width,
            ),
        )

        headers = [
            "Body",
            "Mass (kg)",
            "X (m)",
            "Y (m)",
            "Vx (m/s)",
            "Vy (m/s)",
        ]

        for column, header in enumerate(headers):
            ttk.Label(
                table,
                text=header,
                font=("TkDefaultFont", 9, "bold"),
            ).grid(row=0, column=column, padx=6, pady=6, sticky="w")

        self.body_entries = []

        for i in range(n):
            defaults = (
                "1.0e20",
                str(-5.0e5 if i == 0 else (5.0e5 if i == 1 else 0.0)),
                "0.0",
                "0.0",
                str(20.0 if i == 0 else (-20.0 if i == 1 else 0.0)),
            )

            entries = []

            ttk.Label(table, text=str(i + 1)).grid(
                row=i + 1, column=0, padx=6, pady=4, sticky="w"
            )

            for column, default in enumerate(defaults, start=1):
                entry = ttk.Entry(table, width=18)
                entry.insert(0, default)
                entry.grid(
                    row=i + 1,
                    column=column,
                    padx=6,
                    pady=4,
                    sticky="ew",
                )
                entries.append(entry)

            self.body_entries.append(entries)

    def _read_configuration(self):
        name = self.name_var.get().strip()
        if not name:
            raise ValueError("Experiment name cannot be empty.")

        n = int(self.n_bodies_var.get())
        if n < 2:
            raise ValueError("Number of bodies must be at least 2.")

        duration = float(self.duration_var.get())
        dt = float(self.dt_var.get())

        if duration <= 0:
            raise ValueError("Duration must be greater than zero.")
        if dt <= 0:
            raise ValueError("Timestep must be greater than zero.")

        if duration % dt != 0:
            raise ValueError("Duration must be an exact multiple of timestep.")

        bodies = []

        for index, entries in enumerate(self.body_entries, start=1):
            values = [float(entry.get()) for entry in entries]
            mass, x, y, vx, vy = values

            if mass <= 0:
                raise ValueError(f"Body {index}: mass must be positive.")

            bodies.append(
                {
                    "mass": mass,
                    "x": x,
                    "y": y,
                    "vx": vx,
                    "vy": vy,
                }
            )

        return ExperimentConfiguration(
            name=name,
            scenario=self.scenario_var.get(),
            n_bodies=n,
            duration=duration,
            integrator=self.integrator_var.get(),
            dt=dt,
            bodies=bodies,
        )

    def _start_experiment(self):
        try:
            configuration = self._read_configuration()
        except (ValueError, TypeError, tk.TclError) as exc:
            self.status_var.set("CONFIGURATION ERROR")
            messagebox.showerror("Invalid configuration", str(exc))
            return

        self.app.session.configuration = configuration
        self.status_var.set("EXPERIMENT STARTING")

        # SimulationEngine integration comes in the next GUI milestone.
        self.app.show_page("LivePage")

    def on_show(self):
        self.status_var.set("READY")


    
    def _on_scenario_selected(self, _event=None):
        """Apply the selected scenario's default configuration."""

        scenario = self.scenario_var.get()

        if scenario == "Earth-Sun (perihelion)":
            self._apply_earth_sun_preset()

        elif scenario == "Three-Body Figure Eight":
            self._apply_figure_eight_preset()

        else:
            self._apply_custom_defaults()

    def _apply_custom_defaults(self):
        """Restore the existing generic two-body configuration."""

        self.name_var.set("Two Body Validation")
        self.n_bodies_var.set(2)
        self.duration_var.set("1000")
        self.integrator_var.set("Velocity Verlet")
        self.dt_var.set("1.0")

        self._rebuild_body_table()

    def _apply_earth_sun_preset(self):
        """Load a reproducible, idealized Earth-Sun initial condition."""

        self.name_var.set("Earth-Sun 365-Day Baseline")
        self.n_bodies_var.set(2)
        self.duration_var.set("31536000")
        self.integrator_var.set("Velocity Verlet")
        self.dt_var.set("43200")

        self._rebuild_body_table()

        # Body 1: Sun
        sun = self.body_entries[0]
        sun_values = (
            "1.98847e30",
            "-441800",
            "0",
            "0",
            "-0.091",
        )

        # Body 2: Earth near perihelion
        earth = self.body_entries[1]
        earth_values = (
            "5.9722e24",
            "147097600000",
            "0",
            "0",
            "30286",
        )

        for entries, values in (
            (sun, sun_values),
            (earth, earth_values),
        ):
            for entry, value in zip(entries, values):
                entry.delete(0, tk.END)
                entry.insert(0, value)

        self.status_var.set(
            "EARTH-SUN PRESET LOADED"
        )

    
    def _apply_figure_eight_preset(self):
        """
    Load a scaled equal-mass three-body figure-eight initial condition.

    The dimensionless initial conditions are scaled using:
        length_scale = 1e7 m
        mass_scale = 1e20 kg
        velocity_scale = sqrt(G * mass_scale / length_scale)
    """

        from math import sqrt
        from psamathe.physics.gravity import G

        mass_scale = 1.0e20
        length_scale = 1.0e7
        velocity_scale = sqrt(
            G * mass_scale / length_scale
        )

        self.name_var.set("Three-Body Figure Eight")
        self.n_bodies_var.set(3)
        self.duration_var.set("2450000")
        self.integrator_var.set("Velocity Verlet")
        self.dt_var.set("1000")

        self._rebuild_body_table()

        # Normalized figure-eight initial conditions.
        normalized_bodies = [
            (
            -0.97000436,
            0.24308753,
            0.466203685,
            0.43236573,
            ),
            (
            0.97000436,
            -0.24308753,
            0.466203685,
            0.43236573,
            ),
            (
            0.0,
            0.0,
            -0.93240737,
            -0.86473146,
            ),
        ]

        for entries, (x, y, vx, vy) in zip(
            self.body_entries,
            normalized_bodies,
        ):
            values = (
            f"{mass_scale:.8e}",
            f"{x * length_scale:.8e}",
            f"{y * length_scale:.8e}",
            f"{vx * velocity_scale:.8e}",
            f"{vy * velocity_scale:.8e}",
            )

            for entry, value in zip(entries, values):
                entry.delete(0, tk.END)
                entry.insert(0, value)

        self.status_var.set(
            "THREE-BODY FIGURE-EIGHT PRESET LOADED"
        )

