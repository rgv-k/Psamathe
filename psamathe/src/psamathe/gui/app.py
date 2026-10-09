import tkinter as tk
from tkinter import ttk

from psamathe.gui.pages.config_page import ConfigPage
from psamathe.gui.pages.live_page import LivePage
from psamathe.gui.pages.results_page import ResultsPage
from psamathe.gui.state import ExperimentSession


class PsamatheApp(tk.Tk):
    """Main Psamathe Phase 1 GUI application."""

    def __init__(self):
        super().__init__()

        self.title("Psamathe — Phase 1")
        self.geometry("1100x720")
        self.minsize(900, 600)

        self.session = ExperimentSession()

        self._configure_style()

        container = ttk.Frame(self, padding=18)
        container.pack(fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.pages = {}

        for page_class in (ConfigPage, LivePage, ResultsPage):
            page = page_class(container, self)
            self.pages[page_class.__name__] = page
            page.grid(row=0, column=0, sticky="nsew")

        self.show_page("ConfigPage")

    def _configure_style(self):
        style = ttk.Style(self)

        # Keep the interface native and dependency-free.
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Title.TLabel", font=("TkDefaultFont", 20, "bold"))
        style.configure("Heading.TLabel", font=("TkDefaultFont", 13, "bold"))
        style.configure("Telemetry.TLabel", font=("TkFixedFont", 12))
        style.configure("Primary.TButton", padding=(16, 8))
        style.configure("Danger.TButton", padding=(16, 8))

    def show_page(self, page_name: str):
        page = self.pages[page_name]
        page.tkraise()
        if hasattr(page, "on_show"):
            page.on_show()


def main():
    app = PsamatheApp()
    app.mainloop()


if __name__ == "__main__":
    main()
