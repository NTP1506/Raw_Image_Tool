import sys
import tkinter as tk
import os
sys.path.insert(0, os.path.dirname(__file__))

from ui.main_window import MainWindow


def main():
    root = tk.Tk()
    root.minsize(900, 700)
    app = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
