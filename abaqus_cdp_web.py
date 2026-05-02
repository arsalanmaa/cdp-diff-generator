import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Concrete Impact & CDP Tool", page_icon="⚙️", layout="wide")

# ==========================================
# CUSTOM CSS STYLING
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit default menu & footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Premium Headers */
    h1 {
        color: #0f172a;
        font-weight: 800;
        letter-spacing: -1px;
    }
    h2, h3 {
        color: #1e293b;
        font-weight: 600;
    }
    
    /* Improve Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 2px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 12px;
        padding-bottom: 12px;
        font-weight: 600;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        border-bottom: 2px solid #2563eb !important;
    }

    /* Button Animations */
    .stButton>button {
        border-radius: 6px;
        transition: all 0.2s ease-in-out;
        font-weight: 600;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Custom Info Box */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. CORE MATH FUNCTIONS (DIF)
# ==========================================

def calculate_compression_dif(f_cm, strain_rate, ec0dot, f_cmo):
    alpha_s = 1 / (5 + 9 * (f_cm / f_cmo))
    gamma_s = 10 ** (6.156 * alpha_s - 2)
    
    if strain_rate <= 30:
        dif = (strain_rate / ec0dot) ** (1.026 * alpha_s)
    else:
        dif = gamma_s * (strain_rate / ec0dot) ** (1/3)
    return dif

def calculate_tension_dif(f_cm, strain_rate, ect0, f_cmo):
    alpha_st = 1 / (10 + (6 * (f_cm / f_cmo)))
    gamma_st = 10 ** ((7.112 * alpha_st) - 2.33)
    
    if strain_rate <= 10:
        dif = (strain_rate / ect0) ** (1.016 * alpha_st)
    else:
        dif = gamma_st * (strain_rate / ect0) ** (1/3)
    return dif

# ==========================================
# 2. CORE MATH FUNCTIONS (CDP & Plot)
# ==========================================

def get_plot_curves(E, S_cu, e_cu, e_60, Alpha, S_tu, e_end, Beta, Tension_Recovery, Compression_Recovory, is_meter, Ref_Length):
    E = E * 1000.
    if e_cu <= S_cu/E or e_cu >= 2*S_cu/E:
        st.error(f"Ultimate crushing strain should be in the range [{S_cu/E} ~ {2*S_cu/E}]")
        return None
    if e_end <= S_tu/E:
        st.error(f"End strain should be larger than [{S_tu/E}]")
        return None
    
    # Compression part
    S_c0 = 2*S_cu - E*e_cu
    e_0 = S_c0/E
    e_c_lin = np.linspace(0.0, e_0, num=20, endpoint=False)
    e_para = np.linspace(e_0, e_cu, num=20, endpoint=False)
    S_c_lin = E*e_c_lin
    S_c_para = -((S_cu-S_c0)/(e_cu-e_0)**2)*(e_para-e_0)*(e_para-e_0-2*(e_cu-e_0)) + S_c0

    e_wb_end = e_cu + e_60*np.power(-np.log(0.001/0.99), 1/Alpha)
    e_wb = np.linspace(e_cu, e_wb_end, 50, endpoint=True)
    S_c_wb = S_cu*(0.99*np.exp(-np.power((e_wb-e_cu)/e_60, Alpha)) + 0.01)

    e_c_total = np.concatenate((e_c_lin, e_para, e_wb), axis=None)
    S_c_total = np.concatenate((S_c_lin, S_c_para, S_c_wb), axis=None)

    # Tensile part
    e_t0 = S_tu/E
    e_t_lin = np.linspace(0.0, e_t0, num=20, endpoint=False)
    S_t_lin = E*e_t_lin
    e_t_power = np.linspace(e_t0, e_end, num=50, endpoint=True)
    S_t_power = S_tu*(np.power(np.abs((e_end-e_t_power)/(e_end-e_t0)), Beta))
    
    e_t_total = np.concatenate((e_t_lin, e_t_power), axis=None)
    S_t_total = np.concatenate((S_t_lin, S_t_power), axis=None)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    
    axs[0].plot(e_c_total, S_c_total, label='Compression', marker='o', markersize=5, linestyle='-', color='#2563eb', linewidth=2)
    axs[0].set_xlabel('Strain (ε)', fontsize=11, fontweight='bold')
    axs[0].set_ylabel('Stress (MPa)', fontsize=11, fontweight='bold')
    axs[0].ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
    axs[0].set_title("Compressive Stress-Strain", fontsize=13, color='#1e293b', pad=15)
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend(frameon=True, shadow=True)
    
    axs[1].plot(e_t_total, S_t_total, label='Tension', marker='s', markersize=5, linestyle='-', color='#dc2626', linewidth=2)
    axs[1].set_xlabel('Strain (ε)', fontsize=11, fontweight='bold')
    axs[1].set_ylabel('Stress (MPa)', fontsize=11, fontweight='bold')
    axs[1].ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
    axs[1].set_title("Tensile Stress-Strain", fontsize=13, color='#1e293b', pad=15)
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].legend(frameon=True, shadow=True)
    
    plt.tight_layout()
    return fig

def generate_macro_and_data(E, S_cu, e_cu, e_60, Alpha, S_tu, e_end, Beta, Tension_Recovery, Compression_Recovory, is_meter, Ref_Length, model_name, mat_name):
    E_raw = E * 1000.
    if e_cu <= S_cu/E_raw or e_cu >= 2*S_cu/E_raw:
        return None, None, None
    if e_end <= S_tu/E_raw:
        return None, None, None

    # Compression 
    S_c0 = 2*S_cu - E_raw*e_cu
    e_0 = S_c0/E_raw
    e_c_lin = np.linspace(0.0, e_0, num=20, endpoint=False)
    e_para = np.linspace(e_0, e_cu, num=20, endpoint=False)
    S_c_lin = E_raw*e_c_lin
    S_c_para = -((S_cu-S_c0)/(e_cu-e_0)**2)*(e_para-e_0)*(e_para-e_0-2*(e_cu-e_0)) + S_c0

    e_wb_end = e_cu + e_60*np.power(-np.log(0.001/0.99), 1/Alpha)
    e_wb = np.linspace(e_cu, e_wb_end, 50, endpoint=True)
    S_c_wb = S_cu*(0.99*np.exp(-np.power((e_wb-e_cu)/e_60, Alpha)) + 0.01)

    e_c_total = np.concatenate((e_c_lin, e_para, e_wb), axis=None)
    S_c_total = np.concatenate((S_c_lin, S_c_para, S_c_wb), axis=None)
    Compression_SS_plot=np.stack((e_c_total, S_c_total), axis=1)

    # Tension
    e_t0 = S_tu/E_raw
    e_t_lin = np.linspace(0.0, e_t0, num=20, endpoint=False)
    S_t_lin = E_raw*e_t_lin
    e_t_power = np.linspace(e_t0, e_end, num=50, endpoint=True)
    S_t_power = S_tu*(np.power(np.abs((e_end-e_t_power)/(e_end-e_t0)), Beta))
    
    e_t_total = np.concatenate((e_t_lin, e_t_power), axis=None)
    S_t_total = np.concatenate((S_t_lin, S_t_power), axis=None)
    Tensile_SS_plot=np.stack((e_t_total, S_t_total), axis=1)

    # Abaqus Prep
    e_c_pla = np.concatenate((e_para, e_wb), axis=None)
    S_c_pla = np.concatenate((S_c_para, S_c_wb), axis=None)
    e_c_inelastic = e_c_pla - S_c_pla/E_raw

    indices = np.where(e_c_pla <= e_cu)
    e_cu_index = indices[0][-1] if indices[0].size > 0 else None
    dc = 1.0 - S_c_pla/S_cu
    dc[:e_cu_index] = 0

    Compression_SS = np.stack((S_c_pla, e_c_inelastic), axis=1)
    Compression_D = np.stack((dc, e_c_inelastic), axis=1)

    e_t_pla = e_t_power
    S_t_pla = S_t_power
    mask = S_t_power > 0.01 * S_tu
    e_t_pla = e_t_power[mask]
    S_t_pla = S_t_power[mask]
    
    e_t_cracking = e_t_pla - S_t_pla/E_raw
    u_t_cracking = e_t_cracking*Ref_Length
    dt = 1.0 - S_t_pla/S_tu

    Tensile_SS = np.stack((S_t_pla, u_t_cracking), axis=1)
    Tensile_D = np.stack((dt, u_t_cracking), axis=1)

    Density = 2.4E-9
    Poisson = 0.2

    if is_meter:
        Density = Density*1E12
        E_raw = E_raw*1E6
        Compression_SS[:, 0] *= 1E6
        Tensile_SS[:, 0] *= 1E6
        Compression_SS_plot[:, 1] *= 1E6
        Tensile_SS_plot[:, 1] *= 1E6

    # Generate Python string
    py_script = f"""# Abaqus Python Script to automatically build the CDP Material
# Generated by Unified Concrete Impact & CDP Tool (Web Version)

from abaqus import *
from abaqusConstants import *
import material

model_name = '{model_name}'
mat_name = '{mat_name}'

if model_name not in mdb.models:
    print('ERROR: Model {{}} not found in Abaqus database.'.format(model_name))
else:
    myModel = mdb.models[model_name]
    if mat_name in myModel.materials:
        del myModel.materials[mat_name]  # Overwrite if exists

    myMat = myModel.Material(name=mat_name)
    myMat.Density(table=(({Density}, ), ))
    myMat.Elastic(table=(({E_raw}, {Poisson}), ))
    myMat.ConcreteDamagedPlasticity(table=((40.0, 0.1, 1.16, 0.66667, 0.001), ))

    comp_ss = {Compression_SS.tolist()}
    myMat.concreteDamagedPlasticity.ConcreteCompressionHardening(table=comp_ss)

    tens_ss = {Tensile_SS.tolist()}
    myMat.concreteDamagedPlasticity.ConcreteTensionStiffening(table=tens_ss, type=DISPLACEMENT)

    comp_d = {Compression_D.tolist()}
    myMat.concreteDamagedPlasticity.ConcreteCompressionDamage(table=comp_d, tensionRecovery={Tension_Recovery})

    tens_d = {Tensile_D.tolist()}
    myMat.concreteDamagedPlasticity.ConcreteTensionDamage(table=tens_d, type=DISPLACEMENT, compressionRecovery={Compression_Recovory})

    print('SUCCESS: Material {{}} has been created in model {{}}!'.format(mat_name, model_name))
"""
    
    # Generate CSV strings
    comp_txt = "Strain, Stress\n" + "\n".join([f"{row[0]:.6e},{row[1]:.6e}" for row in Compression_SS_plot])
    tens_txt = "Strain, Stress\n" + "\n".join([f"{row[0]:.6e},{row[1]:.6e}" for row in Tensile_SS_plot])
    
    return py_script, comp_txt, tens_txt


# ==========================================
# 3. STREAMLIT UI
# ==========================================

st.title("🏛️ Concrete Impact & CDP Generation Tool")
st.markdown("### *Advanced web-based utility for dynamic engineering simulations*")
st.divider()

# Initialize session state for wizard steps and dynamic values
if 'step' not in st.session_state: st.session_state.step = 0

# Persistent data storage across wizard steps (prevents garbage collection issues)
if 'data_E' not in st.session_state: st.session_state.data_E = 30.0
if 'data_S_cu' not in st.session_state: st.session_state.data_S_cu = 50.0
if 'data_e_cu' not in st.session_state: st.session_state.data_e_cu = 0.003
if 'data_e_60' not in st.session_state: st.session_state.data_e_60 = 0.005
if 'data_Alpha' not in st.session_state: st.session_state.data_Alpha = 2.0
if 'data_T_Rec' not in st.session_state: st.session_state.data_T_Rec = 1.0

if 'data_S_tu' not in st.session_state: st.session_state.data_S_tu = 5.0
if 'data_e_end' not in st.session_state: st.session_state.data_e_end = 0.002
if 'data_Beta' not in st.session_state: st.session_state.data_Beta = 2.0
if 'data_C_Rec' not in st.session_state: st.session_state.data_C_Rec = 0.0
if 'data_Ref_L' not in st.session_state: st.session_state.data_Ref_L = 1.0

# Navigation callbacks
def next_step():
    st.session_state.step += 1
def prev_step():
    st.session_state.step -= 1

# STEP 0: Introduction
if st.session_state.step == 0:
    st.header("📖 Model Background & Introduction")
    st.info("💡 **Note:** This tool is an advanced extension of the original **CoMat** tool developed by Lim Youngbin (DOI: [10.13140/RG.2.2.25509.10725](https://doi.org/10.13140/RG.2.2.25509.10725)). While the original CoMat generates simple CDP tension and compression curves, this extended version integrates **Dynamic Increase Factor (DIF)** strain rate calculations, dynamic stress derivation, and rigorous damage evolution verification.")
    
    st.markdown("""
    ### Abstract
    Concrete Damaged Plasticity (CDP) is widely used in numerical simulations to model the complex behavior of concrete under various loading conditions. This unified open-source tool automates the generation of CDP data for Abaqus by calculating the Dynamic Increase Factor (DIF) based on strain rate, determining peak compression and tension stresses, and systematically generating both CDP compression and tension curves alongside their respective damage evolution curves.

    ### Theoretical Background
    **1. Dynamic Increase Factor (DIF)**
    The strength of concrete under dynamic loading is typically higher than under quasi-static loading. The unified tool calculates the strain-rate dependent DIF for both compression and tension to construct the full stress-strain relationships.

    **2. Stress-Strain Curve Generation**
    *   **Compression:** Constructed using an initial linear elastic part, a parabolic ascending part up to the peak stress, and a Weibull descending (softening) part.
    *   **Tension:** Constructed using a linear elastic ascending part followed by a power-law descending part.

    **3. Damage Evolution and Plastic Strain**
    The tool derives inelastic/cracking strains and damage parameters ($d_c$, $d_t$) mathematically. It provides rigorous diagnostic checks to ensure the generated plastic strains are positive and monotonically increasing—a strict requirement for Abaqus convergence.
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🚀 Launch Tool", type="primary", use_container_width=True):
            next_step()

# STEP 1: DIF
elif st.session_state.step == 1:
    st.header("Step 1: Dynamic Increase Factor (DIF)")
    col1, col2 = st.columns(2)
    with col1:
        f_cm = st.number_input("Static Compressive Strength (f_cm) [MPa]:", value=45.0)
        comp_rate = st.number_input("Compressive Strain Rate [s^-1]:", value=100.0)
        f_cmo = st.number_input("Ref. Compressive Strength (f_cmo) [MPa]:", value=10.0)
        ec0dot = st.number_input("Ref. Compressive Strain Rate (ec0dot) [s^-1]:", value=30e-6, format="%.2e")
    with col2:
        f_ctm = st.number_input("Static Tensile Strength (f_ctm) [MPa]:", value=3.8)
        tens_rate = st.number_input("Tensile Strain Rate [s^-1]:", value=50.0)
        ect0 = st.number_input("Ref. Tensile Strain Rate (ect0) [s^-1]:", value=3e-6, format="%.2e")

    if st.button("Calculate DIF", type="primary"):
        dif_comp = calculate_compression_dif(f_cm, comp_rate, ec0dot, f_cmo)
        impact_comp = dif_comp * f_cm
        dif_tens = calculate_tension_dif(f_cm, tens_rate, ect0, f_cmo)
        impact_tens = dif_tens * f_ctm
        
        st.session_state.data_S_cu = float(round(impact_comp, 2))
        st.session_state.data_S_tu = float(round(impact_tens, 2))
        
        st.success("DIF Calculated! The dynamic strengths have been saved for the next step.")
        c1, c2 = st.columns(2)
        c1.metric("Dynamic Compressive Strength", f"{impact_comp:.2f} MPa", f"DIF: {dif_comp:.4f}")
        c2.metric("Dynamic Tensile Strength", f"{impact_tens:.2f} MPa", f"DIF: {dif_tens:.4f}")
    
    st.divider()
    c_prev, c_next = st.columns(2)
    with c_prev:
        st.button("⬅️ Previous: Intro", on_click=prev_step)
    with c_next:
        st.button("Next: CDP Parameters ➡️", on_click=next_step, type="primary")

# STEP 2: CDP Parameters
elif st.session_state.step == 2:
    st.header("Step 2: CDP Material Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Compression")
        E = st.number_input("E (GPa):", value=st.session_state.data_E, step=1.0)
        S_cu = st.number_input("σ_cu (MPa):", value=st.session_state.data_S_cu, step=1.0)
        e_cu = st.number_input("ε_cu:", value=st.session_state.data_e_cu, format="%.4f")
        e_60 = st.number_input("ε_0.63:", value=st.session_state.data_e_60, format="%.4f")
        Alpha = st.number_input("α (Compression):", value=st.session_state.data_Alpha)
        T_Rec = st.number_input("w_t (Tension Recovery):", value=st.session_state.data_T_Rec)
        
        st.session_state.data_E = E
        st.session_state.data_S_cu = S_cu
        st.session_state.data_e_cu = e_cu
        st.session_state.data_e_60 = e_60
        st.session_state.data_Alpha = Alpha
        st.session_state.data_T_Rec = T_Rec
        
    with col2:
        st.subheader("Tension")
        S_tu = st.number_input("σ_tu (MPa):", value=st.session_state.data_S_tu, step=0.5)
        e_end = st.number_input("ε_end:", value=st.session_state.data_e_end, format="%.4f")
        Beta = st.number_input("β (Tension):", value=st.session_state.data_Beta)
        C_Rec = st.number_input("w_c (Compression Recovery):", value=st.session_state.data_C_Rec)
        Ref_L = st.number_input("l_ref (Ref. Length):", value=st.session_state.data_Ref_L)
        
        st.session_state.data_S_tu = S_tu
        st.session_state.data_e_end = e_end
        st.session_state.data_Beta = Beta
        st.session_state.data_C_Rec = C_Rec
        st.session_state.data_Ref_L = Ref_L

    st.divider()
    st.subheader("Stress-Strain Plotting")
    if st.button("📊 Plot Stress-Strain Curves", use_container_width=True, type="primary"):
        # Plotting the curves (assumes standard visualization, so is_meter=False for standard viewing)
        fig = get_plot_curves(st.session_state.data_E, st.session_state.data_S_cu, st.session_state.data_e_cu, st.session_state.data_e_60, st.session_state.data_Alpha, st.session_state.data_S_tu, st.session_state.data_e_end, st.session_state.data_Beta, st.session_state.data_T_Rec, st.session_state.data_C_Rec, False, st.session_state.data_Ref_L)
        if fig:
            st.pyplot(fig)

    st.divider()
    c_prev, c_next = st.columns(2)
    with c_prev:
        st.button("⬅️ Previous: DIF", on_click=prev_step)
    with c_next:
        st.button("Next: Damage Verification ➡️", on_click=next_step)

# STEP 3: Verification
elif st.session_state.step == 3:
    st.header("Step 3: Damage Plotting & Verification")
    
    st.subheader("Compression Verification")
    if st.button("Verify & Plot Compression Damage"):
        try:
            E_val = st.session_state.data_E * 1000.0
            S_cu = st.session_state.data_S_cu
            e_cu = st.session_state.data_e_cu
            e_60 = st.session_state.data_e_60
            Alpha = st.session_state.data_Alpha
            
            S_c0 = 2*S_cu - E_val*e_cu
            e_0 = S_c0/E_val

            e_para = np.linspace(e_0, e_cu, num=20, endpoint=False)
            S_c_para = -((S_cu-S_c0)/(e_cu-e_0)**2)*(e_para-e_0)*(e_para-e_0-2*(e_cu-e_0)) + S_c0

            e_wb_end = e_cu + e_60*np.power(-np.log(0.001/0.99), 1/Alpha)
            e_wb = np.linspace(e_cu, e_wb_end, 50, endpoint=True)
            S_c_wb = S_cu*(0.99*np.exp(-np.power((e_wb-e_cu)/e_60, Alpha)) + 0.01)

            e_c = np.concatenate((e_para, e_wb), axis=None)
            S_c = np.concatenate((S_c_para, S_c_wb), axis=None)

            e_el = S_c / E_val
            e_in = e_c - e_el
            
            dc = np.zeros_like(e_c)
            peak_idx = np.argmin(np.abs(e_c - e_cu))
            dc[peak_idx:] = 1.0 - (S_c[peak_idx:] / S_cu)
            dc = np.clip(dc, 0.0, 0.9999) 
            
            e_pl = e_in - (dc / (1.0 - dc)) * e_el

            tol = -1e-10
            is_positive = np.all(e_pl >= tol)
            is_monotonic = np.all(np.diff(e_pl) >= tol)
            
            st.code(f'''--- COMPRESSION DIAGNOSTICS ---
Total Points: {len(e_c)}
Max Damage (d_c): {np.max(dc):.4f}
Max Plastic Strain (ε_pl): {np.max(e_pl):.6e}
-----------------------------------
{"✅ PASSED: Plastic strain is positive." if is_positive else "❌ FAILED: Negative plastic strain!"}
{"✅ PASSED: Strain is monotonically increasing." if is_monotonic else "❌ FAILED: Strain decreases (Abaqus will abort)."}''')
            
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            axs[0].plot(e_c, S_c, 'b-')
            axs[0].set_title('Stress vs Total Strain')
            axs[0].set_xlabel('Strain ε_c')
            axs[0].set_ylabel('Stress σ_c (MPa)')
            axs[0].grid(True)

            axs[1].plot(e_in, dc, 'r-')
            axs[1].set_title('Damage vs Inelastic Strain')
            axs[1].set_xlabel('Inelastic Strain ε_in')
            axs[1].set_ylabel('Damage d_c')
            axs[1].grid(True)

            axs[2].plot(e_in, e_pl, 'g-')
            axs[2].set_title('Plastic Strain vs Inelastic Strain')
            axs[2].set_xlabel('Inelastic Strain ε_in')
            axs[2].set_ylabel('Plastic Strain ε_pl')
            axs[2].grid(True)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")

    st.subheader("Tension Verification")
    if st.button("Verify & Plot Tension Damage"):
        try:
            E_val = st.session_state.data_E * 1000.0
            S_tu = st.session_state.data_S_tu
            e_end = st.session_state.data_e_end
            Beta = st.session_state.data_Beta
            
            e_t0 = S_tu/E_val
            e_t_power = np.linspace(e_t0, e_end, num=50, endpoint=True)
            S_t_power = S_tu*(np.power(np.abs((e_end-e_t_power)/(e_end-e_t0)), Beta))
            
            mask = S_t_power > 0.01 * S_tu
            e_t = e_t_power[mask]
            S_t = S_t_power[mask]

            e_el = S_t / E_val
            e_ck = e_t - e_el  
            
            dt = 1.0 - (S_t / S_tu)
            dt = np.clip(dt, 0.0, 0.9999) 
            
            e_pl = e_ck - (dt / (1.0 - dt)) * e_el

            tol = -1e-10
            is_positive = np.all(e_pl >= tol)
            is_monotonic = np.all(np.diff(e_pl) >= tol)

            st.code(f'''--- TENSION DIAGNOSTICS ---
Total Cracking Points: {len(e_t)}
Max Tension Damage (d_t): {np.max(dt):.4f}
Max Plastic Strain (ε_pl): {np.max(e_pl):.6e}
-----------------------------------
{"✅ PASSED: Plastic strain is positive." if is_positive else "❌ FAILED: Negative plastic strain!"}
{"✅ PASSED: Strain is monotonically increasing." if is_monotonic else "❌ FAILED: Strain decreases (Abaqus will abort)."}''')
            
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            axs[0].plot(e_t, S_t, 'b-')
            axs[0].set_title('Stress vs Total Strain')
            axs[0].set_xlabel('Strain ε_t')
            axs[0].set_ylabel('Stress σ_t (MPa)')
            axs[0].grid(True)

            axs[1].plot(e_ck, dt, 'r-')
            axs[1].set_title('Damage vs Cracking Strain')
            axs[1].set_xlabel('Cracking Strain ε_ck')
            axs[1].set_ylabel('Damage d_t')
            axs[1].grid(True)

            axs[2].plot(e_ck, e_pl, 'g-')
            axs[2].set_title('Plastic Strain vs Cracking Strain')
            axs[2].set_xlabel('Cracking Strain ε_ck')
            axs[2].set_ylabel('Plastic Strain ε_pl')
            axs[2].grid(True)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    c_prev, c_next = st.columns(2)
    with c_prev:
        st.button("⬅️ Previous: CDP Parameters", on_click=prev_step)
    with c_next:
        st.button("Next: Abaqus Export ➡️", on_click=next_step, type="primary")

# STEP 4: Export
elif st.session_state.step == 4:
    st.header("Step 4: 💾 Abaqus Export Configuration")
    st.markdown("Define the model and material names that will be written into the Abaqus Python Macro script.")
    col_abq1, col_abq2 = st.columns(2)
    with col_abq1:
        model_name = st.text_input("Target Abaqus Model Name:", value="Model-1")
    with col_abq2:
        mat_name = st.text_input("New Material Name:", value="CDP_Dynamic")
    
    st.info("💡 Enable this if you are using SI(m) units in Abaqus. It will automatically scale the outputs.")
    is_meter = st.checkbox("Dimension is in meter (Abaqus Scale)", value=False)
    
    st.divider()
    st.subheader("Final Generation Actions")
    if st.button("💾 Generate Abaqus Script (.py)", type="primary", use_container_width=True):
        py_script, comp_txt, tens_txt = generate_macro_and_data(st.session_state.data_E, st.session_state.data_S_cu, st.session_state.data_e_cu, st.session_state.data_e_60, st.session_state.data_Alpha, st.session_state.data_S_tu, st.session_state.data_e_end, st.session_state.data_Beta, st.session_state.data_T_Rec, st.session_state.data_C_Rec, is_meter, st.session_state.data_Ref_L, model_name, mat_name)
        
        if py_script:
            st.success("Generation Successful! Download your files below.")
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            with dl_col1:
                st.download_button(label="Download Macro (.py)", data=py_script, file_name="Create_CDP_Material.py", mime="text/x-python", use_container_width=True)
            with dl_col2:
                st.download_button(label="Download Comp Data (.txt)", data=comp_txt, file_name="Compression_SS.txt", mime="text/plain", use_container_width=True)
            with dl_col3:
                st.download_button(label="Download Tens Data (.txt)", data=tens_txt, file_name="Tensile_SS.txt", mime="text/plain", use_container_width=True)

    st.divider()
    st.button("⬅️ Previous: Damage Verification", on_click=prev_step)
