import sys
import tkinter as tk

from ui.main_window import MainWindow

def main():
    root = tk.Tk()
    app = MainWindow(root)

    def on_closing():
        root.quit()
        root.destroy()
        sys.exit()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
