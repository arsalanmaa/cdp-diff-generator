# Concrete Impact & CDP Generation Tool

An advanced, web-based engineering utility for dynamic simulations. This tool automates the generation of Concrete Damaged Plasticity (CDP) data for Abaqus. 

It acts as an extension of the original CoMat tool, integrating Dynamic Increase Factor (DIF) strain rate calculations, dynamic stress derivation, and rigorous damage evolution verification to ensure your plastic strains are positive and monotonically increasing.

## Features
- Dynamic Increase Factor (DIF) computation based on strain rates.
- CDP Material Parameter generation for both compression and tension.
- Instant Stress-Strain plotting.
- Rigorous Abaqus diagnostics and Damage verification.
- Automated generation of Abaqus Python Macro scripts (`.py`) and data files (`.txt`).

## How to Run Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run abaqus_cdp_web.py`
