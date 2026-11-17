# ASTRONET-TESS // Exoplanet Transit Detection & Astrophysical Vetting Observatory

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Tri--Branch%201D--CNN-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20Telemetry-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React_18-Mission%20Control%20UI-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-Multi--Container-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![ESA Gaia DR3](https://img.shields.io/badge/ESA%20Gaia-DR3%20Astrometry-1B365D?style=flat-square)](https://www.cosmos.esa.int/web/gaia/dr3)

**ASTRONET-TESS** is an end-to-end, multi-modal astronomical vetting workstation engineered to detect, characterize, and validate transiting exoplanet candidates from **NASA TESS (Transiting Exoplanet Survey Satellite)** space-telescope photometry.

Space-based transit surveys suffer from high false-positive rates caused by grazing eclipsing binaries (EBs), hierarchical triple star systems, and background eclipsing binaries (BEBs) blended within TESS's wide $21''/\text{pixel}$ detector footprint. ASTRONET-TESS addresses this end-to-end by uniting **iterative Box Least Squares (BLS) signal peeling**, **raw CCD Target Pixel File (TPF) centroid tracking**, **ESA Gaia DR3 astrometric flux dilution correction**, and a **Tri-Branch Hybrid 1D-CNN + Physics MLP** classifier inside a containerized Mission Control web interface.

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

Validated Benchmark Targets

1. The Mission Control launch deck includes four pre-configured astrophysical benchmark systems:
2. L 98-59 — Compact multi-planet terrestrial system ideal for testing iterative multi-planet BLS peeling (Depth: 2–3 Planets).
3. TOI-700 — Nearby M-dwarf system hosting an Earth-sized planet inside the habitable zone.
4. WASP-46 — High-SNR Hot Jupiter transit demonstrating deep U-shaped transit morphology and clean on-target CCD centroid lock.
5. TIC 53593457 — Known astrophysical false positive benchmark used to verify eclipsing binary / centroid shift rejection.

## Table of Contents

- [System Pipeline & Data Flow](#system-pipeline--data-flow)
- [Scientific & Astrophysical Methodology](#scientific--astrophysical-methodology)
- [Tri-Branch Hybrid Deep Learning Architecture](#tri-branch-hybrid-deep-learning-architecture)
- [Observatory Interface Showcase](#observatory-interface-showcase)
- [Project Architecture](#project-architecture)
- [Quick Start (Docker Compose)](#quick-start-docker-compose)
- [Local Development Setup](#local-development-setup)
- [REST API Reference](#rest-api-reference)
- [Validated Benchmark Targets](#validated-benchmark-targets)
- [Author](#author)

---

## System Pipeline & Data Flow

```text
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                         NASA MAST & ESA GAIA DR3 ARCHIVES                            │
│   • Light Curve FITS (PDCSAP / SAP)    • Raw Target Pixel Files    • Gaia DR3 Cone   │
└───────────────────┬─────────────────────────────────┬───────────────────────┬────────┘
                    │                                 │                       │
                    ▼                                 ▼                       ▼
┌───────────────────────────────────────┐   ┌───────────────────┐   ┌──────────────────┐
│  1. PHOTOMETRIC PREPROCESSING         │   │ 2. CCD PIXEL AUDIT│   │ 3. GAIA DILUTION │
│  • 5σ Outlier Clipping                │   │ • Aperture Mask   │   │ • 42" Cone Query │
│  • Savitzky-Golay Variability Flatten │   │ • In/Out Centroid │   │ • Flux Blend %   │
│  • Multi-Sector Stitching (Optional)  │   │ • 21"/px Sky Shift│   │ • True Rp Solver │
└───────────────────┬───────────────────┘   └─────────┬─────────┘   └─────────┬────────┘
                    │                                 │                       │
                    ▼                                 │                       │
┌───────────────────────────────────────┐             │                       │
│  4. ITERATIVE BLS SIGNAL PEELING      │             │                       │
│  • Periodogram Scan [P_min, P_max]    │             │                       │
│  • Phase Folding & Cadence Parity     │             │                       │
│  • Transit Masking for Multi-Planets  │             │                       │
└───────────────────┬───────────────────┘             │                       │
                    │                                 │                       │
                    ▼                                 │                       │
┌───────────────────────────────────────┐             │                       │
│  5. TRI-BRANCH HYBRID PYTORCH MODEL   │             │                       │
│  • Global 1D-CNN (201 bins)           │             │                       │
│  • Local 1D-CNN  (61 bins)            │             │                       │
│  • Physics MLP   (5 Scalars)          │             │                       │
│  • Gradient × Input Attribution       │             │                       │
└───────────────────┬───────────────────┘             │                       │
                    └─────────────────────────────────┼───────────────────────┘
                                                      ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                   FASTAPI TELEMETRY SERVER + SQLITE LOGGING (:8000)                  │
└─────────────────────────────────────────────────────┬────────────────────────────────┘
                                                      │ JSON REST + CORS
                                                      ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                REACT + VITE + PLOTLY.JS MISSION CONTROL WORKSTATION (:5173)          │
│  • Dynamic AI Synthesis  • Multi-Planet Table  • Interactive Plots  • CSV/PNG Export │
└──────────────────────────────────────────────────────────────────────────────────────┘

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



Author
Syiyamkumar S
B.E. Computer Science • Sathyabama Institute of Science and Technology, Chennai
GitHub: @Syiyamkumar
