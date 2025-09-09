# OptiLab – Advanced Optical Properties Calculator  
*A Python GUI tool that turns your UV-Vis-NIR spectra into publication-ready optical constants.*

---

##  What it does
1. Load spectra from **CSV** or **Excel**  
2. Auto-complete missing **T/R/A** columns (enforces T + A + R = 1)  
3. Compute **α, k, n, ε′, ε″, tan δ** for **thin films** or **solutions**  
4. Interactive **Tauc & Urbach** plots with click-drag linear fitting  
5. Export everything (data + summary) to a multi-sheet **Excel** file

---

##  Optical quantities extracted
| Symbol | Quantity | Unit | Note |
|--------|----------|------|------|
| α      | absorption coefficient | cm⁻¹ | thickness corrected |
| k      | extinction coefficient | – | k = αλ / 4π |
| n      | refractive index | – | K-K + Cauchy approx. |
| ε′     | real dielectric constant | – | ε′ = n² – k² |
| ε″     | dielectric loss | – | ε″ = 2nk |
| tan δ  | loss tangent | – | tan δ = ε″/ε′ |
| Eg     | optical band gap | eV | Tauc fit ± σ |
| Eu     | Urbach energy | eV | Urbach fit ± σ |

---

##  Quick start
1. Install requirements  
   ```bash
   pip install -r requirements.txt
   # core: pandas, numpy, matplotlib, scipy, openpyxl