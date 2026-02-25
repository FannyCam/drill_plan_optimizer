import tkinter as tk
from tkinter import filedialog, messagebox
import json
import numpy as np
import os
from algo.drill_model import DrillModel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Drill Hole Optimizer")
        self.root.geometry("400x300")
        self.boundary = None
        self.bootlegs = None

        self.option_frame = tk.Frame(root)
        self.option_frame.pack(pady=10, padx=20, anchor="w")

        self.btn_load = tk.Button(self.option_frame, text="Load JSON", command=self.load_file)
        self.btn_load.pack(side=tk.LEFT) 

        self.label_file = tk.Label(self.option_frame, text="No file loaded", fg="grey", font=("Arial", 10, "italic"))
        self.label_file.pack(side=tk.LEFT, padx=10)

        self.opti_frame = tk.Frame(root)
        self.opti_frame.pack(pady=10, padx=20, anchor="w")

        self.btn_opti = tk.Button(self.opti_frame, text="Run Optimization", command=self.optimize, state=tk.DISABLED)
        self.btn_opti.pack(side=tk.RIGHT) 

        self.chart_frame = tk.Frame(root)
        self.chart_frame.pack(fill=tk.BOTH, expand=True)

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Choose a JSON file",
            filetypes=[("JSON files", "*.json")]
        )
        
        if path is None:
            return None

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if not isinstance(data, dict):
                    return self._handle_error("Invalid JSON: Root must be an object {}.")
                
                temp_boundary = data.get("boundary")
                temp_bootlegs = data.get("bootlegs")
    
                if not isinstance(temp_boundary, list) or not isinstance(temp_bootlegs, list):
                    return self._handle_error("Data format error: 'boundary' and 'bootlegs' must be lists.")
                
                return data, path
            
        except json.JSONDecodeError:
            return self._handle_error("Invalid JSON format. Check for syntax errors.")
        except KeyError as e:
            return self._handle_error(str(e))
        except Exception as e:
            return self._handle_error(f"Unexpected error: {str(e)}")

    def _handle_error(self, message):
        messagebox.showerror("Error", message)
        self.label_file.config(text="Load failed", fg="#c0392b") 
        return None

    def load_file(self):
        result = self._open_file()

        if result is None:
            return
        
        data, path = result

        self.boundary = np.array(data.get("boundary", []))
        self.bootlegs = np.array(data.get("bootlegs", []))
        
        filename = os.path.basename(path)
        self.label_file.config(text=f"{filename}", fg="grey", font=("Arial", 10, "italic", "bold"))
        self.btn_opti.config(state=tk.NORMAL)

    def optimize(self):
        drill_model = DrillModel(self.boundary, self.bootlegs)
        drill_model.optimize()

        fig = drill_model.plot()
        
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
            
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, self.chart_frame)
        toolbar.update()
        
        canvas.draw()
        plt.close(fig)