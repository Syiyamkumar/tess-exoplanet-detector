# ASTRONET-TESS // Exoplanet Transit Detection & Astrophysical Vetting Observatory

Multi-modal exoplanet transit vetting observatory combining PyTorch Tri-Branch 1D-CNN, iterative BLS periodograms, TPF CCD centroid audits, ESA Gaia DR3 flux dilution, FastAPI, and React.

**ASTRONET-TESS** is an end-to-end, multi-modal astronomical vetting workstation engineered to detect and validate transiting exoplanet candidates from **NASA TESS (Transiting Exoplanet Survey Satellite)** photometry.

By uniting iterative **Box Least Squares (BLS)** signal extraction, **raw CCD Target Pixel File (TPF)** photocenter centroid auditing, **ESA Gaia DR3** astrometric flux dilution correction, and a **Tri-Branch Hybrid 1D-CNN + Physics MLP** classifier, the platform separates genuine planetary transits from astrophysical false positives such as eclipsing binaries (EBs) and background eclipsing binaries (BEBs).

---

## Key Scientific & Engineering Capabilities

1. **Iterative Multi-Planet BLS Signal Peeling (`src/api.py`, `src/preprocess.py`)**
   - Ingests live light curves from **NASA MAST** (by star name, TIC ID, or RA/Dec coordinates) or custom `.FITS` / `.CSV` uploads.
   - Applies Savitzky-Golay stellar variability detrending and $5\sigma$ outlier clipping.
   - Extracts periodic signals via **Box Least Squares (BLS)** periodograms, masks out detected transits, and iteratively re-scans residual photometry to uncover multi-planet systems (e.g., **L 98-59**, **TOI-700**).
   - Includes a **Synthetic Transit Injection-Recovery** module to benchmark pipeline sensitivity.

2. **Tri-Branch Hybrid Deep Learning Architecture (`src/model.py`, `src/train.py`)**
   - **Global Photometric Branch (1D-CNN)**: Processes 201-bin phase-folded light curves ($\phi \in [-0.5, 0.5]$) to detect secondary eclipses and out-of-transit ellipsoidal variations.
   - **Local Transit Morphology Branch (1D-CNN)**: Processes 61-bin zoomed transit windows to distinguish flat-bottomed U-shaped planetary transits from V-shaped grazing stellar eclipses.
   - **Astrophysical Scalar Branch (MLP)**: Integrates 5 domain-engineered physical features—transit depth (ppm), odd-vs-even transit depth ratio, secondary eclipse significance, host stellar radius ($R_\odot$), and derived planetary radius ($R_\oplus$).
   - Achieves **0.843 out-of-sample ROC-AUC** across 1,100+ NASA ExoFOP TESS targets, paired with **Gradient $\times$ Input Explainable AI** feature attribution.

3. **Raw CCD Pixel Centroid Offset Audit (`src/api.py`)**
   - Downloads raw **Target Pixel Files (TPFs)** ($21''/\text{pixel}$ plate scale) from MAST.
   - Computes sub-pixel photocenter coordinates during transit vs. out-of-transit baseline to detect angular centroid shifts ($> 2.5''$) caused by blended background eclipsing binaries.

4. **ESA Gaia DR3 Cone-Search & Flux Dilution Correction (`src/gaia_vetting.py`)**
   - Queries the **ESA Gaia DR3** catalog within a $42''$ radius around target coordinates via `astroquery`.
   - Quantifies aperture flux contamination from neighboring stars and solves for the true undiluted planetary radius:
     $$R_{p,\text{true}} = R_{p,\text{obs}} \sqrt{1 + \frac{F_{\text{contaminants}}}{F_{\text{target}}}}$$

5. **Mission Control React Workstation (`frontend/`)**
   - Built with **React 18**, **Vite**, **Tailwind CSS**, and **Plotly.js** featuring an editorial deep-burgundy, warm-cream, and copper observatory aesthetic.
   - Supports dynamic natural-language candidate synthesis, interactive multi-planet row selection, hoverable astrophysical glossary tooltips, one-click **CSV / PNG** telemetry export, and a persistent **SQLite** vetting history log.

---

## Project Architecture

```text
tess-exoplanet-detector/
├── .gitignore
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── docs/
│   └── screenshots/                        # Observatory UI captures
├── data/
│   ├── raw/                                # Local MAST FITS cache (gitignored)
│   ├── processed/                          # Binned phase arrays (gitignored)
│   ├── toicatalog.csv                      # NASA ExoFOP TESS Object of Interest catalog
│   └── vetting_history.db                  # Persistent SQLite vetting log (gitignored)
├── models/
│   ├── exoplanet_hybrid_cnn.pt             # Trained PyTorch Tri-Branch Hybrid CNN weights
│   └── tabular_scaler.joblib               # Fitted StandardScaler for astrophysical scalars
├── src/
│   ├── preprocess.py                       # Phase-folding, binning & vetting feature extraction
│   ├── model.py                            # Tri-Branch Hybrid 1D-CNN + MLP PyTorch architecture
│   ├── train.py                            # Training & ROC-AUC validation pipeline
│   ├── gaia_vetting.py                     # ESA Gaia DR3 cone search & flux dilution solver
│   └── api.py                              # FastAPI telemetry server & SQLite logger
└── frontend/
    ├── Dockerfile
    ├── .dockerignore
    ├── package.json
    ├── vite.config.js
    ├── index.html
    ├── public/
    │   └── cosmos-bg.jpg                   # Custom astronomical backdrop
    └── src/
        ├── main.jsx
        ├── index.css
        ├── App.jsx                         # Mission Control router & state manager
        └── components/
            ├── InfoTooltip.jsx             # Interactive astrophysical glossary popovers
            ├── AstronomicalArtwork.jsx     # Ringed exoplanet SVG & standby transit schematic
            ├── Navbar.jsx                  # Sticky observatory navigation bar
            ├── MissionHero.jsx             # Editorial mission brief & benchmark launch deck
            ├── ObservatoryView.jsx         # Multi-planet table, KPI cards & 2x2 Plotly suite
            ├── PixelAuditView.jsx          # 2D CCD heatmap & sub-pixel centroid scatter
            └── GaiaDilutionView.jsx        # Gaia DR3 sky field map & dilution table
