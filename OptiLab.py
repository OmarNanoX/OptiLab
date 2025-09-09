# (Full file — updated sections include fit_tauc_region, fit_urbach_region, and display of results)
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
from scipy.stats import linregress
from scipy.interpolate import interp1d
import matplotlib.patches as patches

class OpticalToolGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Optical Properties Calculator")
        self.filename = None
        self.glass_ref_data = None
        self.span_selector = None
        self.selected_range = None
        self.current_plot_type = None

        # Variables
        self.sample_type = tk.StringVar(value="Solution")
        self.thickness = tk.StringVar(value="100")
        self.path_length = tk.StringVar(value="1.0")  # cm for solutions
        self.correct_glass_abs = tk.BooleanVar(value=False)
        self.tauc_type = tk.StringVar(value="Direct allowed")
        self.refractive_index_medium = tk.StringVar(value="1.0")

        # Results storage
        self.results = {}

        # GUI Layout
        self.setup_widgets()

    def setup_widgets(self):
        # Main control frame
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill="x")

        # File operations
        ttk.Button(control_frame, text="Upload CSV/XLSX", command=self.load_file).grid(row=0, column=0, padx=5, pady=5)

        # Sample type
        ttk.Label(control_frame, text="Sample Type:").grid(row=0, column=1, padx=5)
        ttk.Combobox(control_frame, textvariable=self.sample_type, values=["Solution", "Thin Film"], 
                    state="readonly", width=12).grid(row=0, column=2, padx=5)

        # Thickness/Path length
        ttk.Label(control_frame, text="Thickness (nm) / Path (cm):").grid(row=0, column=3, padx=5)
        thickness_frame = ttk.Frame(control_frame)
        thickness_frame.grid(row=0, column=4, padx=5)
        ttk.Entry(thickness_frame, textvariable=self.thickness, width=8).pack(side="left")
        ttk.Entry(thickness_frame, textvariable=self.path_length, width=8).pack(side="left", padx=(2,0))

        # Refractive index of medium
        ttk.Label(control_frame, text="n_medium:").grid(row=0, column=5, padx=5)
        ttk.Entry(control_frame, textvariable=self.refractive_index_medium, width=8).grid(row=0, column=6, padx=5)

        # Second row of controls
        control_frame2 = ttk.Frame(self.root, padding=5)
        control_frame2.pack(fill="x")

        # Tauc plot type
        ttk.Label(control_frame2, text="Tauc Type:").grid(row=0, column=0, padx=5)
        ttk.Combobox(control_frame2, textvariable=self.tauc_type, 
                    values=["Direct allowed", "Indirect allowed", "Direct forbidden", "Indirect forbidden"],
                    state="readonly", width=15).grid(row=0, column=1, padx=5)

        # Action buttons
        ttk.Button(control_frame2, text="Plot Raw Data", command=self.plot_raw_data).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame2, text="Calculate All Properties", command=self.calculate_all_properties).grid(row=0, column=3, padx=5)
        ttk.Button(control_frame2, text="Show Tauc Plot", command=self.show_tauc_plot).grid(row=0, column=4, padx=5)
        ttk.Button(control_frame2, text="Show Urbach Plot", command=self.show_urbach_plot).grid(row=0, column=5, padx=5)
        ttk.Button(control_frame2, text="Export Results", command=self.export_results).grid(row=0, column=6, padx=5)

        # Instructions
        instruction_frame = ttk.Frame(self.root, padding=5)
        instruction_frame.pack(fill="x")
        self.instruction_label = ttk.Label(instruction_frame, 
                                         text="Instructions: Upload data → Calculate Properties → Select regions on Tauc/Urbach plots for fitting",
                                         foreground="blue")
        self.instruction_label.pack()

        # Plot frame
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Create notebook for multiple plots
        self.notebook = ttk.Notebook(plot_frame)
        self.notebook.pack(fill="both", expand=True)

        # Main plot tab
        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="Main Plot")
        
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Results tab
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="Results")
        
        # Results display
        self.results_text = tk.Text(self.results_tab, wrap=tk.WORD, font=("Courier", 10))
        scrollbar = ttk.Scrollbar(self.results_tab, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def load_file(self):
        self.filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
        if not self.filename:
            return
        try:
            if self.filename.endswith(".csv"):
                self.data = pd.read_csv(self.filename)
            elif self.filename.endswith(".xlsx"):
                self.data = pd.read_excel(self.filename)

            # Check for required columns
            if "Wavelength" not in self.data.columns:
                raise ValueError("File must contain 'Wavelength' column.")

            # Fill missing optical properties
            self.fill_missing_TAR()
            
            messagebox.showinfo("Success", f"Loaded {self.filename} successfully.\nColumns: {list(self.data.columns)}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def fill_missing_TAR(self):
        """Fill missing Transmittance, Absorbance, Reflectance values"""
        df = self.data
        cols = df.columns
        
        # Convert percentages to fractions if needed
        for col in ["Absorbance", "Transmittance", "Reflectance"]:
            if col in cols and df[col].max() > 1:
                df[col] = df[col] / 100
        
        # Fill missing values based on conservation: T + A + R = 1
        if all(k in cols for k in ["Absorbance", "Transmittance"]):
            df["Reflectance"] = (1 - df["Absorbance"] - df["Transmittance"]).clip(lower=0, upper=1)
        elif all(k in cols for k in ["Absorbance", "Reflectance"]):
            df["Transmittance"] = (1 - df["Absorbance"] - df["Reflectance"]).clip(lower=0, upper=1)
        elif all(k in cols for k in ["Transmittance", "Reflectance"]):
            df["Absorbance"] = (1 - df["Transmittance"] - df["Reflectance"]).clip(lower=0, upper=1)
        
        # Ensure all values are between 0 and 1
        for col in ["Absorbance", "Transmittance", "Reflectance"]:
            if col in df.columns:
                df[col] = df[col].clip(lower=0, upper=1)


    def plot_raw_data(self):
        if not hasattr(self, 'data'):
            messagebox.showwarning("No Data", "Please upload data first.")
            return

        self.ax.clear()
        self.current_plot_type = "raw"
        
        # Plot available data
        if "Absorbance" in self.data.columns:
            self.ax.plot(self.data["Wavelength"], self.data["Absorbance"], 'r-', label="Absorbance", linewidth=2)
        if "Transmittance" in self.data.columns:
            self.ax.plot(self.data["Wavelength"], self.data["Transmittance"], 'g-', label="Transmittance", linewidth=2)
        if "Reflectance" in self.data.columns:
            self.ax.plot(self.data["Wavelength"], self.data["Reflectance"], 'b-', label="Reflectance", linewidth=2)

        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Intensity")
        self.ax.set_title(f"Raw Optical Data - {self.sample_type.get()}")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def calculate_all_properties(self):
        if not hasattr(self, 'data'):
            messagebox.showwarning("No Data", "Please upload data first.")
            return

        try:
            self.fill_missing_TAR()

            # Basic parameters
            wl_nm = self.data["Wavelength"]  # nm
            wl_m = wl_nm * 1e-9  # meters
            
            # Physical constants
            h = 6.626e-34  # Planck constant
            c = 3e8        # Speed of light
            e = 1.602e-19  # Elementary charge
            
            # Photon energy
            hv = h * c / wl_m / e  # eV

            # Calculate Absorption Coefficient α based on sample type
            if self.sample_type.get() == "Thin Film":
                # For thin films, use T and R
                if "Transmittance" not in self.data.columns or "Reflectance" not in self.data.columns:
                    raise ValueError("For thin films, both Transmittance and Reflectance are required")
                
                T = self.data["Transmittance"]
                R = self.data["Reflectance"]
                d_nm = float(self.thickness.get())  # thickness in nm
                
                # Beer-Lambert law for thin films: T = (1-R)²*exp(-αd)
                # Solving for α: α = -ln(T/(1-R)²)/d
                # Handle case where R approaches 1 or T approaches 0
                T_eff = np.maximum(T, 1e-10)  # Avoid log(0)
                R_eff = np.minimum(R, 0.999)  # Avoid division by 0
                
                # Absorption coefficient in nm^-1
                alpha = -np.log(T_eff / ((1 - R_eff)**2)) / d_nm
                alpha = np.maximum(alpha, 0)  # Ensure non-negative values
                    
            else:  # Solution
                # For solutions, use A only
                if "Absorbance" not in self.data.columns:
                    raise ValueError("For solutions, Absorbance is required")
                
                A = self.data["Absorbance"]
                path_cm = float(self.path_length.get())  # path length in cm
                
                # Beer-Lambert law: A = αcl where c*l = path_cm
                # Convert to nm^-1: α = A * ln(10) / (path_cm * 1e7)  # cm to nm conversion
                alpha = A * np.log(10) / (path_cm * 1e7)  # nm^-1

            # Convert α from nm^-1 to cm^-1 for conventional units
            alpha_cm = alpha * 1e7  # nm^-1 to cm^-1 (for convention)

            # 2. Extinction coefficient k
            k = alpha * wl_nm / (4 * np.pi)  # dimensionless (using α in nm^-1)

            # 3. Refractive index n (using Kramers-Kronig relations approximation)
            n_medium = float(self.refractive_index_medium.get())
            
            # For thin films with interference effects
            if self.sample_type.get() == "Thin Film" and "Reflectance" in self.data.columns:
                R = self.data["Reflectance"]
                # Simplified calculation for refractive index from reflectance
                # R = ((n-1)²+k²)/((n+1)²+k²) - approximate for small k
                # Solving approximately: n ≈ (1+√R)/(1-√R) * n_medium
                R_eff = np.clip(R, 0.001, 0.999)  # Avoid division issues
                sqrt_R = np.sqrt(R_eff)
                n = n_medium * (1 + sqrt_R) / (1 - sqrt_R)
            else:
                # For solutions or when detailed calculation isn't possible
                # Use Cauchy dispersion relation as approximation: n = A + B/λ²
                # Or simply use medium refractive index with small correction
                n = n_medium + k  # Simple approximation

            # 4. Dielectric constant ε' (real part)
            epsilon_real = n**2 - k**2

            # 5. Dielectric loss ε'' (imaginary part)  
            epsilon_imag = 2 * n * k

            # 6. Dielectric loss tangent tan δ
            tan_delta = np.where(epsilon_real != 0, epsilon_imag / epsilon_real, 0)

            # Store results
            self.data["Photon_Energy_eV"] = hv
            self.data["Alpha_nm-1"] = alpha  # nm^-1
            self.data["Alpha_cm-1"] = alpha_cm  # cm^-1 (conventional)
            self.data["Extinction_k"] = k
            self.data["Refractive_n"] = n
            self.data["Epsilon_real"] = epsilon_real
            self.data["Epsilon_imag"] = epsilon_imag
            self.data["Tan_delta"] = tan_delta

            # Store summary results
            self.results = {
                "Sample Type": self.sample_type.get(),
                "Thickness/Path": f"{self.thickness.get()} nm" if self.sample_type.get() == "Thin Film" else f"{self.path_length.get()} cm",
                "Wavelength Range": f"{wl_nm.min():.1f} - {wl_nm.max():.1f} nm",
                "Photon Energy Range": f"{hv.min():.2f} - {hv.max():.2f} eV",
                "Max Absorption Coefficient": f"{alpha_cm.max():.2e} cm⁻¹",
                "Max Absorption Coefficient (nm^-1)": f"{alpha.max():.2e} nm⁻¹",
                "Max Extinction Coefficient": f"{k.max():.4f}",
                "Refractive Index Range": f"{n.min():.2f} - {n.max():.2f}",
                "Max Dielectric Constant": f"{epsilon_real.max():.2f}",
                "Max Dielectric Loss": f"{epsilon_imag.max():.4f}",
                "Max Loss Tangent": f"{tan_delta.max():.4f}"
            }

            self.display_results()
            messagebox.showinfo("Success", "All optical properties calculated successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))


    def show_tauc_plot(self):
        if not hasattr(self, 'data') or "Alpha_cm-1" not in self.data.columns:
            messagebox.showwarning("No Data", "Please calculate properties first.")
            return

        self.ax.clear()
        self.current_plot_type = "tauc"

        alpha = self.data["Alpha_cm-1"]
        hv = self.data["Photon_Energy_eV"]

        # Calculate Tauc plot based on type
        tauc_type = self.tauc_type.get()
        if tauc_type == "Direct allowed":
            tauc_y = (alpha * hv)**2
            ylabel = "(αhν)²"
        elif tauc_type == "Indirect allowed":
            tauc_y = (alpha * hv)**0.5
            ylabel = "(αhν)^(1/2)"
        elif tauc_type == "Direct forbidden":
            tauc_y = (alpha * hv)**(2/3)
            ylabel = "(αhν)^(2/3)"
        elif tauc_type == "Indirect forbidden":
            tauc_y = (alpha * hv)**(1/3)
            ylabel = "(αhν)^(1/3)"

        self.data["Tauc_Y"] = tauc_y

        self.ax.plot(hv, tauc_y, 'bo-', markersize=4, linewidth=1.5, label=f"Tauc Plot ({tauc_type})")
        self.ax.set_xlabel("Photon Energy (eV)")
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(f"Tauc Plot - {tauc_type}\nSelect linear region to calculate band gap")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()

        # Setup span selector for region selection
        self.setup_span_selector()
        self.canvas.draw()

    def show_urbach_plot(self):
        if not hasattr(self, 'data') or "Alpha_cm-1" not in self.data.columns:
            messagebox.showwarning("No Data", "Please calculate properties first.")
            return

        self.ax.clear()
        self.current_plot_type = "urbach"

        alpha = self.data["Alpha_cm-1"]
        hv = self.data["Photon_Energy_eV"]

        # Urbach plot: ln(α) vs hν
        ln_alpha = np.log(alpha.replace(0, np.nan)).fillna(0) if isinstance(alpha, pd.Series) else np.log(alpha)
        self.data["ln_Alpha"] = ln_alpha

        self.ax.plot(hv, ln_alpha, 'ro-', markersize=4, linewidth=1.5, label="Urbach Plot")
        self.ax.set_xlabel("Photon Energy (eV)")
        self.ax.set_ylabel("ln(α)")
        self.ax.set_title("Urbach Plot\nSelect linear region to calculate Urbach energy")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()

        # Setup span selector for region selection
        self.setup_span_selector()
        self.canvas.draw()

    def setup_span_selector(self):
        # Remove existing span selector
        if self.span_selector:
            self.span_selector.disconnect_events()
        
        # Create new span selector
        self.span_selector = SpanSelector(
            self.ax, 
            self.on_select_range, 
            'horizontal',
            useblit=True,
            props=dict(alpha=0.5, facecolor='red'),
            interactive=True
        )

    def on_select_range(self, xmin, xmax):
        """Handle range selection for fitting"""
        self.selected_range = (xmin, xmax)
        
        if self.current_plot_type == "tauc":
            self.fit_tauc_region(xmin, xmax)
        elif self.current_plot_type == "urbach":
            self.fit_urbach_region(xmin, xmax)

    def fit_tauc_region(self, xmin, xmax):
        """Fit linear region in Tauc plot to find band gap, with uncertainty propagation"""
        try:
            hv = self.data["Photon_Energy_eV"].values
            tauc_y = self.data["Tauc_Y"].values
            
            # Select data in range
            mask = (hv >= xmin) & (hv <= xmax)
            x_fit = hv[mask]
            y_fit = tauc_y[mask]
            
            if len(x_fit) < 3:
                messagebox.showwarning("Selection Error", "Please select a larger range with more data points.")
                return
            
            # Use numpy.polyfit with covariance to get parameter uncertainties
            p, cov = np.polyfit(x_fit, y_fit, 1, cov=True)
            slope = p[0]
            intercept = p[1]
            # covariance matrix: cov[0,0]=var_slope, cov[1,1]=var_intercept, cov[0,1]=cov(slope,intercept)
            var_slope = cov[0,0]
            var_intercept = cov[1,1]
            cov_slope_intercept = cov[0,1]
            slope_stderr = np.sqrt(var_slope)
            intercept_stderr = np.sqrt(var_intercept)
            
            # Compute R^2 from residuals
            y_pred = slope * x_fit + intercept
            ss_res = np.sum((y_fit - y_pred)**2)
            ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
            r_squared = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0
            
            # Band gap (x-intercept where y = 0)
            if slope != 0:
                band_gap = -intercept / slope
                # Propagate uncertainty: g = -b/m
                # dg/db = -1/m ; dg/dm = b/m^2
                dg_db = -1.0 / slope
                dg_dm = intercept / (slope**2)
                var_bandgap = (dg_db**2)*var_intercept + (dg_dm**2)*var_slope + 2*dg_db*dg_dm*cov_slope_intercept
                band_gap_err = np.sqrt(var_bandgap) if var_bandgap >= 0 else np.nan
            else:
                band_gap = 0.0
                band_gap_err = np.nan
            
            # Plot fit line
            x_extended = np.linspace(min(x_fit), max(max(x_fit), band_gap + 0.5), 200)
            y_fit_line = slope * x_extended + intercept
            
            # Clear and replot
            self.ax.clear()
            hv_all = self.data["Photon_Energy_eV"]
            tauc_y_all = self.data["Tauc_Y"]
            self.ax.plot(hv_all, tauc_y_all, 'bo-', markersize=4, linewidth=1.5, label=f"Tauc Plot ({self.tauc_type.get()})")
            self.ax.plot(x_fit, y_fit, 'ro', markersize=6, label="Selected Region")
            self.ax.plot(x_extended, y_fit_line, 'r--', linewidth=2, label=f"Linear Fit (R² = {r_squared:.4f})")
            if not np.isnan(band_gap):
                self.ax.axvline(x=band_gap, color='green', linestyle=':', linewidth=2, label=f"Band Gap = {band_gap:.3f} ± {band_gap_err:.3f} eV")
            
            self.ax.set_xlabel("Photon Energy (eV)")
            ylabel = {"Direct allowed": "(αhν)²", "Indirect allowed": "(αhν)^(1/2)", 
                     "Direct forbidden": "(αhν)^(2/3)", "Indirect forbidden": "(αhν)^(1/3)"}[self.tauc_type.get()]
            self.ax.set_ylabel(ylabel)
            self.ax.set_title(f"Tauc Plot - {self.tauc_type.get()}\nBand Gap = {band_gap:.3f} ± {band_gap_err:.3f} eV")
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            
            # Store result (with uncertainty)
            self.results[f"Band Gap ({self.tauc_type.get()})"] = (band_gap, band_gap_err)
            self.results["Tauc Fit R²"] = f"{r_squared:.4f}"
            # Also store slope/intercept with uncertainties for traceability
            self.results["Tauc Fit Slope"] = (slope, slope_stderr)
            self.results["Tauc Fit Intercept"] = (intercept, intercept_stderr)
            
            self.canvas.draw()
            self.display_results()
            
        except Exception as e:
            messagebox.showerror("Fitting Error", str(e))

    def fit_urbach_region(self, xmin, xmax):
        """Fit linear region in Urbach plot to find Urbach energy, with uncertainty propagation"""
        try:
            hv = self.data["Photon_Energy_eV"].values
            ln_alpha = self.data["ln_Alpha"].values
            
            # Select data in range
            mask = (hv >= xmin) & (hv <= xmax)
            x_fit = hv[mask]
            y_fit = ln_alpha[mask]
            
            if len(x_fit) < 3:
                messagebox.showwarning("Selection Error", "Please select a larger range with more data points.")
                return
            
            # Use numpy.polyfit with covariance to get parameter uncertainties
            p, cov = np.polyfit(x_fit, y_fit, 1, cov=True)
            slope = p[0]
            intercept = p[1]
            var_slope = cov[0,0]
            var_intercept = cov[1,1]
            cov_slope_intercept = cov[0,1]
            slope_stderr = np.sqrt(var_slope)
            intercept_stderr = np.sqrt(var_intercept)
            
            # Compute R^2
            y_pred = slope * x_fit + intercept
            ss_res = np.sum((y_fit - y_pred)**2)
            ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
            r_squared = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0
            
            # Urbach energy (1/|slope|)
            if slope != 0:
                urbach_energy = 1.0 / abs(slope)
                # Propagate uncertainty: U = 1/m  => dU/dm = -1/m^2  (we use absolute value in reporting)
                # So variance = (1/m^4) * var_slope
                urbach_err = slope_stderr / (slope**2)
                # take absolute for display
                urbach_err = abs(urbach_err)
            else:
                urbach_energy = float('inf')
                urbach_err = np.nan
            
            # Plot fit line
            x_extended = np.linspace(xmin, xmax, 100)
            y_fit_line = slope * x_extended + intercept
            
            # Clear and replot
            self.ax.clear()
            hv_all = self.data["Photon_Energy_eV"]
            ln_alpha_all = self.data["ln_Alpha"]
            self.ax.plot(hv_all, ln_alpha_all, 'ro-', markersize=4, linewidth=1.5, label="Urbach Plot")
            self.ax.plot(x_fit, y_fit, 'bo', markersize=6, label="Selected Region")
            self.ax.plot(x_extended, y_fit_line, 'b--', linewidth=2, label=f"Linear Fit (R² = {r_squared:.4f})")
            
            self.ax.set_xlabel("Photon Energy (eV)")
            self.ax.set_ylabel("ln(α)")
            self.ax.set_title(f"Urbach Plot\nUrbach Energy = {urbach_energy:.4f} ± {urbach_err:.4f} eV")
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            
            # Store result (with uncertainty)
            self.results["Urbach Energy"] = (urbach_energy, urbach_err)
            self.results["Urbach Fit R²"] = f"{r_squared:.4f}"
            self.results["Urbach Fit Slope"] = (slope, slope_stderr)
            self.results["Urbach Fit Intercept"] = (intercept, intercept_stderr)
            
            self.canvas.draw()
            self.display_results()
            
        except Exception as e:
            messagebox.showerror("Fitting Error", str(e))

    def display_results(self):
        """Display results in the results tab"""
        self.results_text.delete(1.0, tk.END)
        
        header = "="*60 + "\n"
        header += "OPTICAL PROPERTIES CALCULATION RESULTS\n"
        header += "="*60 + "\n\n"
        
        self.results_text.insert(tk.END, header)
        
        for key, value in self.results.items():
            # If value is tuple (val, err) display ±
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], (int, float, np.floating)):
                val, err = value
                line = f"{key:.<30} {val:.4f} ± {err:.4f}\n"
            else:
                line = f"{key:.<30} {value}\n"
            self.results_text.insert(tk.END, line)
        
        if hasattr(self, 'data') and len(self.data) > 0:
            self.results_text.insert(tk.END, "\n" + "="*60 + "\n")
            self.results_text.insert(tk.END, "DETAILED DATA STATISTICS\n")
            self.results_text.insert(tk.END, "="*60 + "\n\n")
            
            # Statistical summary of calculated properties
            for col in ["Alpha_cm-1", "Extinction_k", "Refractive_n", "Epsilon_real", "Epsilon_imag", "Tan_delta"]:
                if col in self.data.columns:
                    data_col = self.data[col]
                    stats = f"{col}:\n"
                    stats += f"  Mean: {data_col.mean():.4e}\n"
                    stats += f"  Max:  {data_col.max():.4e}\n"
                    stats += f"  Min:  {data_col.min():.4e}\n\n"
                    self.results_text.insert(tk.END, stats)

    def export_results(self):
        """Export results to Excel file"""
        if not hasattr(self, 'data'):
            messagebox.showwarning("No Data", "No data to export.")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
            )
            
            if filename:
                if filename.endswith('.xlsx'):
                    # Create Excel file with multiple sheets
                    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                        # Main data
                        self.data.to_excel(writer, sheet_name='Optical_Data', index=False)
                        
                        # Results summary
                        # Convert tuple results to string for export
                        results_df = pd.DataFrame([
                            (k, f"{v[0]:.6g} ± {v[1]:.6g}") if isinstance(v, tuple) and len(v)==2 and isinstance(v[0], (int,float,np.floating)) else (k, v) 
                            for k,v in self.results.items()
                        ], columns=['Property', 'Value'])
                        results_df.to_excel(writer, sheet_name='Summary', index=False)
                        
                elif filename.endswith('.csv'):
                    self.data.to_csv(filename, index=False)
                
                messagebox.showinfo("Export Success", f"Data exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    # Placeholder: implement glass correction if you have function previously (kept for compatibility)

if __name__ == '__main__':
    root = tk.Tk()
    root.state('zoomed')  # Maximize window on Windows
    app = OpticalToolGUI(root)
    root.mainloop()
