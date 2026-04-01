"""
Path Simulation App with Comprehensive Validation
A clean interface for running return path simulations with detailed validation displays
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime
import uuid
import zipfile
import io
from pathlib import Path

from simulation_engine import SimulationConfig, PathSimulator
def fig_to_png_bytes(fig, scale: int = 2) -> bytes:
    # Requires: pip install -U kaleido
    return fig.to_image(format="png", scale=scale)


RETURN_LINE_STYLE = dict(color='#ef4444', width=2.5)
SIGNAL_LINE_STYLE = dict(color='#3b82f6', width=2.5, dash='6px,3px')


def compute_shared_y_range(results, config):
    """Use one y-axis range across all path charts."""
    if results.get('returns_pred') is None or results.get('signal_pred') is None:
        return [-20, 20]

    H = config.time_horizon
    T_is = config.years_insample

    values = [
        results['returns_pred'][H:T_is+1:H, :] * 100,
        results['returns_pred'][0:T_is+H+1:H, :] * 100,
        results['signal_pred'][0:T_is+H+1:H, :] * 100,
    ]
    y_all = np.concatenate([arr.reshape(-1) for arr in values if arr.size > 0])
    y_min = np.floor(np.nanmin(y_all) / 10) * 10 - 10
    y_max = np.ceil(np.nanmax(y_all) / 10) * 10 + 10
    return [y_min, y_max]


def add_ten_percent_reference_lines(fig, y_range):
    """Add slightly heavier guide lines at 10% intervals."""
    start = int(np.ceil(y_range[0] / 10.0) * 10)
    end = int(np.floor(y_range[1] / 10.0) * 10)
    for y in range(start, end + 1, 10):
        fig.add_hline(y=y, line_width=1.6, line_color='rgba(0,0,0,0.24)', layer='below')


def apply_path_chart_layout(fig, y_range):
    fig.update_layout(
        width=650,
        height=450,
        title="",
        template='plotly_white',
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis=dict(
            title="Time",
            range=[-40, 0],
            tickmode='linear',
            tick0=-40,
            dtick=1,
            ticklabelstep=5,
            showgrid=True,
            zeroline=False,
            gridcolor='rgba(0,0,0,0.10)'
        ),
        yaxis=dict(
            title="Return",
            range=y_range,
            tickmode='linear',
            tickformat=".0f",
            ticksuffix="%",
            dtick=2,
            ticklabelstep=5,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.12)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.25)'
        )
    )
    add_ten_percent_reference_lines(fig, y_range)


def build_path_figure(results, config, path_index: int, include_realized_return: bool = False):
    """Build a path chart only when the UI needs it."""
    fig = make_subplots()

    H = config.time_horizon
    T_is = config.years_insample

    if include_realized_return:
        r_plot = results['returns_pred'][H:T_is+H+1:H, path_index] * 100
        x_returns = np.arange(-40, 1)
    else:
        r_plot = results['returns_pred'][H:T_is+1:H, path_index] * 100
        x_returns = np.arange(-40, 0)

    s_plot = results['signal_pred'][0:T_is+H+1:H, path_index] * 100
    x_signal = np.arange(-40, 1)

    fig.add_trace(go.Scatter(
        x=x_returns, y=r_plot,
        name='Return',
        line=RETURN_LINE_STYLE
    ))

    fig.add_trace(go.Scatter(
        x=x_signal, y=s_plot,
        name='Predictive signal',
        line=SIGNAL_LINE_STYLE
    ))

    return fig

# Page configuration
st.set_page_config(
    page_title="Path Simulation",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .validation-pass {
        color: #10b981;
        font-weight: bold;
    }
    .validation-fail {
        color: #ef4444;
        font-weight: bold;
    }
    .alert-warning {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
        color: #1f2937;
    }
    .alert-success {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
        color: #1f2937;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'run_id' not in st.session_state:
    st.session_state.run_id = None

# Presets
PRESETS = {
    "Current (2-year)": {
        "time_horizon": 2,
        "target_correlation": 0.50,
        "n_paths_predictable": 30,
        "mu": 0.0607,
        "sigma_eps": 0.192,
        "phi": 0.92,
        "sigma_delta": 0.152,
        "rho": -0.72,
    },
    "Alternative (5-year)": {
        "time_horizon": 5,
        "target_correlation": 0.57,
        "n_paths_predictable": 30,
        "mu": 0.0607,
        "sigma_eps": 0.192,
        "phi": 0.92,
        "sigma_delta": 0.152,
        "rho": -0.72,
    }
}

# App header
st.markdown('<div class="main-header">📈 Path Simulation</div>', unsafe_allow_html=True)

# Sidebar - Parameter inputs
with st.sidebar:
    st.header("Simulation Parameters")
    
    # Preset selector
    preset_choice = st.selectbox(
        "Preset Configuration",
        ["Custom"] + list(PRESETS.keys()),
        help="Select a preset or choose Custom to set your own parameters"
    )
    
    # Load preset if selected
    if preset_choice != "Custom":
        preset_params = PRESETS[preset_choice]
    else:
        preset_params = PRESETS["Current (2-year)"]  # Default values
    
    st.divider()
    
    # Path Configuration
    st.subheader("Path Configuration")
    
    time_horizon = st.number_input(
        "Time Horizon (years)",
        min_value=1,
        max_value=10,
        value=preset_params["time_horizon"],
        help="Returns are averaged over H years"
    )
    

    n_paths_predictable = st.number_input(
        "Predictable Paths",
        min_value=1,
        max_value=50,
        value=preset_params["n_paths_predictable"],
        help="Set to 0 to skip predictable paths"
    )
    
    st.divider()
    
    # Correlation Settings
    st.subheader("Correlation Settings")
    
    target_correlation = st.slider(
        "Target Correlation",
        min_value=0.0,
        max_value=1.0,
        value=preset_params["target_correlation"],
        step=0.01,
        help=f"For {time_horizon}-year returns. R² ≈ {preset_params['target_correlation']**2:.3f}"
    )
    
    correlation_window = st.slider(
        "Correlation Window",
        min_value=0.01,
        max_value=0.10,
        value=0.02,
        step=0.01,
        help="Accept correlations within ± this amount"
    )
    
    st.divider()
    
    # Advanced Parameters (collapsible)
    with st.expander("Advanced Parameters"):
        st.caption("Return Parameters")
        mu = st.number_input(
            "Mean Return (μ)",
            value=preset_params["mu"],
            format="%.4f",
            help="Mean annual arithmetic return"
        )
        
        sigma_eps = st.number_input(
            "Return Volatility (σ_ε)",
            value=preset_params["sigma_eps"],
            format="%.4f",
            help="Annual arithmetic return volatility"
        )
        
        st.caption("Signal Parameters (AR(1))")
        phi = st.slider(
            "Persistence (φ)",
            min_value=0.0,
            max_value=0.99,
            value=preset_params["phi"],
            step=0.01
        )
        
        sigma_delta = st.number_input(
            "Signal Volatility (σ_δ)",
            value=preset_params["sigma_delta"],
            format="%.4f"
        )
        
        rho = st.slider(
            "Shock Correlation (ρ)",
            min_value=-1.0,
            max_value=1.0,
            value=preset_params["rho"],
            step=0.01,
            help="Correlation between signal and return shocks"
        )
        
        st.caption("Simulation Settings")
        n_simulations = st.number_input(
            "Number of Simulations",
            min_value=100,
            max_value=50000,
            value=10000,
            step=1000,
            help="Generate this many paths before selecting the best"
        )
        
        random_seed = st.number_input(
            "Random Seed",
            min_value=1,
            max_value=9999,
            value=42,
            help="For reproducibility"
        )
    
    st.divider()
    
    # Validation settings
    st.subheader("Validation Settings")
    
    # SPX Data Path
    spx_path = st.text_input(
        "S&P 500 Data Path (optional)",
        value="",
        help="Path to SPX_5Y_Returns.xlsx for validation"
    )
    
    # Check if SPX data exists
    default_spx = Path(__file__).parent.parent.parent / "Data" / "data used in original code" / "SPX_5Y_Returns.xlsx"
    spx_available = (spx_path and Path(spx_path).exists()) or default_spx.exists()
    
    # Validation toggle
    enable_validation = st.checkbox(
        "Enable SPX Validation",
        value=spx_available,  # Default to True if SPX data exists
        help="Enable validation against S&P 500 historical data",
        disabled=not spx_available,  # Disable checkbox if no SPX data found
        key="enable_validation_checkbox"
    )
    
    # Store in session state for debugging
    st.session_state.enable_validation = enable_validation
    
    # Show path info
    if spx_path and Path(spx_path).exists():
        st.info(f"✓ Using custom SPX data: {spx_path}")
    elif default_spx.exists():
        st.info(f"✓ Using default SPX data: {default_spx}")
    else:
        st.warning("⚠️ No SPX data found. Validation disabled. Provide path above to enable.")
    
    st.divider()
    
    # Run button
    run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

    st.divider()
    
    # Export Controls
    st.subheader("Results Export")
    show_sample_paths = st.checkbox(
        "Render Sample Path Charts",
        value=False,
        help="Leave this off for faster runs. Plotly chart rendering is one of the slowest UI steps."
    )
    st.session_state.show_sample_paths = show_sample_paths

    enable_png_export = st.checkbox(
        "Enable PNG Plot Export",
        value=False,
        help="Prepare downloadable PNG plot bundles only when needed. This can be slow."
    )
    st.session_state.enable_png_export = enable_png_export

    enable_csv_export = st.checkbox(
        "Enable CSV Export of Paths",
        value=False,
        help="Generate a CSV file containing the full return paths for all simulations"
    )
    st.session_state.enable_csv_export = enable_csv_export
    
    enable_plots_export = st.checkbox(
        "Enable Plots Export (HTML)",
        value=False,
        help="Generate interactive HTML plots for all sample paths"
    )
    st.session_state.enable_plots_export = enable_plots_export

# Main content area
if run_button:
    with st.spinner("Running simulation..."):
        # Determine SPX path and validation setting
        spx_data_path = None
        validate_spx = False
        
        if enable_validation:
            if spx_path and Path(spx_path).exists():
                spx_data_path = spx_path
                validate_spx = True
            else:
                # Try default path
                default_path = Path(__file__).parent.parent.parent / "Data" / "data used in original code" / "SPX_5Y_Returns.xlsx"
                if default_path.exists():
                    spx_data_path = str(default_path)
                    validate_spx = True
        
        # Create configuration
        config = SimulationConfig(
            n_paths_predictable=n_paths_predictable,
            time_horizon=time_horizon,
            target_correlation=target_correlation,
            correlation_window=correlation_window,
            mu=mu,
            sigma_eps=sigma_eps,
            phi=phi,
            sigma_delta=sigma_delta,
            rho=rho,
            n_simulations=n_simulations,
            random_seed=random_seed,
            validate_with_spx=validate_spx,
            spx_data_path=spx_data_path,
            save_example_plots=False
        )
        
        # Run simulation
        simulator = PathSimulator(config)
        results = simulator.run()
       
        # Generate run ID
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Store in session state
        st.session_state.results = results
        st.session_state.run_id = run_id
        st.session_state.config = config
        
        st.success("✓ Simulation completed!")
        
        st.rerun()

# Display results
if st.session_state.results is not None:
    results = st.session_state.results
    config = st.session_state.config
    validation = results.get('validation', {})
            
    # Alert/Warning System (PRIORITY 6)
    if validation:
        pred_compliance = validation.get('predictable', {}).get('pass_rates', {}).get('full_compliance', 1.0)
        
        # Calculate R² warnings
        target_r2 = config.target_correlation ** 2
        warnings = []
        
        if 'predictable' in validation:
            pred_regs = validation['predictable'].get('regressions', [])
            if pred_regs:
                mean_r2_signal = np.mean([r['r2_signal'] for r in pred_regs if not np.isnan(r['r2_signal'])])
                mean_r2_lag = np.mean([r['r2_lag'] for r in pred_regs if not np.isnan(r['r2_lag'])])
                
                if abs(mean_r2_signal - target_r2) > 0.10:
                    warnings.append(f"Mean R²(signal) ({mean_r2_signal:.3f}) differs from target ({target_r2:.3f}) by more than 0.10")
                if mean_r2_lag > 0.10:
                    warnings.append(f"Mean R²(lag) ({mean_r2_lag:.3f}) > 0.10 indicates serial correlation problem")
        
        if pred_compliance < 0.7 or warnings:
            st.markdown(f"""
            <div class="alert-warning">
                <strong>⚠️ Validation Warning</strong><br>
                Some paths have low compliance rates (Predictable: {pred_compliance*100:.0f}%).<br>
                {('<br>'.join(warnings) + '<br>') if warnings else ''}
                Consider: (1) Increasing N_SIMULATIONS to {config.n_simulations * 2:,}, or (2) Loosening correlation window to {correlation_window + 0.02:.2f}
            </div>
            """, unsafe_allow_html=True)
        elif pred_compliance >= 0.9:
            st.markdown(f"""
            <div class="alert-success">
                <strong>✓ Excellent Validation</strong><br>
                High compliance rates achieved (Predictable: {pred_compliance*100:.0f}%).
            </div>
            """, unsafe_allow_html=True)
    
    # Configuration Summary (PRIORITY 7)
    with st.expander("📋 Configuration Summary", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Path Settings**")
            st.write(f"Time Horizon: {config.time_horizon} years")
            st.write(f"Predictable Paths: {config.n_paths_predictable}")
        
        with col2:
            st.write("**Correlation**")
            st.write(f"Target: {config.target_correlation:.3f}")
            st.write(f"Window: ±{config.correlation_window:.3f}")
            st.write(f"Target R²: {config.target_correlation**2:.3f}")
        
        with col3:
            st.write("**Simulation**")
            st.write(f"N Simulations: {config.n_simulations:,}")
            st.write(f"Random Seed: {config.random_seed}")
            st.write(f"SPX Validation: {'Yes' if config.validate_with_spx else 'No'}")
        
        st.divider()
        
        # Effective Parameters and Run ID (Request id: 6)
        st.write(f"**Run ID:** `{st.session_state.run_id}`")
        
        with st.expander("🔬 Effective Parameters for this Run"):
            st.code(f"""
Time Horizon (H):       {config.time_horizon}
Target Correlation:     {config.target_correlation}
Path Configuration:     {config.n_paths_predictable} Pred 
Mean Return (mu):       {config.mu}
Return Vol (sigma_eps): {config.sigma_eps}
Persistence (phi):      {config.phi}
Signal Vol (sigma_d):   {config.sigma_delta}
Shock Corr (rho):       {config.rho}
Random Seed:            {config.random_seed}
            """, language="text")
    
    # Summary metrics
    st.header("Simulation Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Time Horizon",
            f"{config.time_horizon} years"
        )
    
    with col2:
        st.metric(
            "Total Paths",
            f"{config.n_paths_predictable}"
        )
    
    with col3:
        if results['correlations_pred'] is not None:
            mean_corr_pred = np.mean(results['correlations_pred'])
            st.metric(
                "Mean Correlation (Pred)",
                f"{mean_corr_pred:.3f}"
            )
        else:
            st.metric("Mean Correlation (Pred)", "N/A")
    
    
    st.divider()
    
    # SPX Baseline Statistics Card (PRIORITY 2)
    if validation and 'spx_stats' in validation:
        st.subheader("📊 S&P 500 Reference Data")
        spx_stats = validation['spx_stats']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**{config.time_horizon}-Year Returns**")
            spx_h = spx_stats['h_year']
            st.write(f"Mean: {spx_h['mean']:.4f} ({spx_h['mean']*100:.2f}%)")
            st.write(f"Std Dev: {spx_h['std']:.4f}")
            st.write(f"Min: {spx_h['min']:.4f} | Max: {spx_h['max']:.4f}")
            st.write(f"Observations: {spx_h['n_obs']}")
        
        with col2:
            st.write("**Annual Returns**")
            spx_1 = spx_stats['annual']
            st.write(f"Mean: {spx_1['mean']:.4f} ({spx_1['mean']*100:.2f}%)")
            st.write(f"Std Dev: {spx_1['std']:.4f}")
            st.write(f"Observations: {spx_1['n_obs']}")
        
        st.divider()

    if validation and 'simulated_stats' in validation:
        st.subheader("📊 Simulated Return Data")
        simulated_stats = validation['simulated_stats']

        col1, col2 = st.columns(2)

        with col1:
            if 'h_year' in simulated_stats:
                st.write(f"**{config.time_horizon}-Year Returns**")
                sim_h = simulated_stats['h_year']
                st.write(f"Mean: {sim_h['mean']:.4f} ({sim_h['mean']*100:.2f}%)")
                st.write(f"Std Dev: {sim_h['std']:.4f}")
                st.write(f"Min: {sim_h['min']:.4f} | Max: {sim_h['max']:.4f}")
                st.write(f"Observations: {sim_h['n_obs']}")

        with col2:
            if 'annual' in simulated_stats:
                st.write("**Annual Returns**")
                sim_1 = simulated_stats['annual']
                st.write(f"Mean: {sim_1['mean']:.4f} ({sim_1['mean']*100:.2f}%)")
                st.write(f"Std Dev: {sim_1['std']:.4f}")
                st.write(f"Min: {sim_1['min']:.4f} | Max: {sim_1['max']:.4f}")
                st.write(f"Observations: {sim_1['n_obs']}")

        st.divider()
    
    # Validation Results Panel (PRIORITY 1)

    # R² Summary Card (HIGH PRIORITY)
    if validation and ('predictable' in validation):
        st.subheader("📈 R² Analysis")
        
        target_r2 = config.target_correlation ** 2
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Target R²:** {target_r2:.3f} (H={config.time_horizon} years)")
            st.write(f"**Theory:** R² = ρ² = {config.target_correlation:.3f}² = {target_r2:.3f}")
        
        with col2:
            st.write("**Interpretation:**")
            st.write("• R²(signal): Variance explained by signal")
            st.write("• R²(lag): Variance explained by past returns")
        
        st.write("")
        
    

        if 'predictable' in validation:
            st.write("**Predictable Paths:**")
            pred_regs = validation['predictable'].get('regressions', [])
            if pred_regs:
                r2_signal_vals = [r['r2_signal'] for r in pred_regs if not np.isnan(r['r2_signal'])]
                r2_lag_vals = [r['r2_lag'] for r in pred_regs if not np.isnan(r['r2_lag'])]
                
                if r2_signal_vals and r2_lag_vals:
                    mean_r2_signal = np.mean(r2_signal_vals)
                    std_r2_signal = np.std(r2_signal_vals)
                    mean_r2_lag = np.mean(r2_lag_vals)
                    std_r2_lag = np.std(r2_lag_vals)
                    
                    # Color coding
                    signal_diff = abs(mean_r2_signal - target_r2)
                    signal_status = "✓" if signal_diff <= 0.05 else ("~" if signal_diff <= 0.10 else "✗")
                    lag_status = "✓" if mean_r2_lag < 0.05 else ("~" if mean_r2_lag < 0.10 else "✗")
                    
                    signal_color = "validation-pass" if signal_diff <= 0.05 else ("" if signal_diff <= 0.10 else "validation-fail")
                    lag_color = "validation-pass" if mean_r2_lag < 0.05 else ("" if mean_r2_lag < 0.10 else "validation-fail")
                    
                    st.markdown(f'<span class="{signal_color}">Signal R²: {mean_r2_signal:.3f} ± {std_r2_signal:.3f} {signal_status}</span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="{lag_color}">Lagged R²: {mean_r2_lag:.3f} ± {std_r2_lag:.3f} {lag_status}</span>', unsafe_allow_html=True)
                else:
                    st.write("Insufficient data for analysis")
            else:
                st.write("No regression data available")
    

        
        st.divider()
    
    if validation and ('predictable' in validation):
        st.header("🔍 Validation Results")
        
        # Determine which columns to show
        has_pred = 'predictable' in validation
        
        if has_pred:
            col1= st.columns(1)[0]
        else:
            col1 = st.container()
        
        
        # Predictable paths column
        if has_pred:
            val_pred = validation['predictable']
            

            st.subheader("Predictable Paths")
            
            # Full compliance
            n_compliant = sum(val_pred.get('full_compliant', []))
            n_total = len(val_pred.get('full_compliant', [1]))
            compliance_rate = n_compliant / n_total if n_total > 0 else 0
            
            st.metric(
                "Full Compliance",
                f"{n_compliant}/{n_total}",
                f"{compliance_rate*100:.0f}%"
            )
            
            # Individual checks
            pass_rates = val_pred.get('pass_rates', {})
            
            checks = [
                ("KS Test (h-year)", pass_rates.get('ks_h_year', 0), 
                    "Tests if h-year return distribution matches S&P 500 historical distribution"),
                ("KS Test (annual)", pass_rates.get('ks_annual', 0),
                    "Tests if annual return distribution matches S&P 500 historical distribution"),
                ("Return Bounds", pass_rates.get('return_bounds', 0),
                    "Checks if min/max returns fall within S&P 500 historical range"),
                ("Volatility Bounds", pass_rates.get('volatility_bounds', 0),
                    "Checks if return volatility is within ±5% of S&P 500 volatility"),
            ]
            
            for check_name, rate, help_text in checks:
                status = "✓" if rate >= 0.7 else "✗"
                color = "validation-pass" if rate >= 0.7 else "validation-fail"
                st.markdown(f'<span class="{color}" title="{help_text}">{status} {check_name}: {rate*100:.0f}% pass</span>', unsafe_allow_html=True)
                st.caption(help_text)
            
            # Autocorrelation
            autocorrs = val_pred.get('autocorrelations', [])
            mean_autocorr = np.nanmean(autocorrs) if autocorrs else 0
            st.write(f"Mean Autocorrelation: {mean_autocorr:.4f} (should be ~0)")
    
        # Detailed validation breakdown
        with st.expander("📋 Detailed Validation by Check"):
            tab1, tab2, tab3, tab4 = st.tabs(["KS Tests", "Bounds", "Volatility", "Autocorrelation"])
            
            with tab1:
                col1= st.columns(1)[0]
                st.write("**Predictable**")
                if 'predictable' in validation:
                    ks_h = validation['predictable'].get('ks_h_pass', [])
                    ks_1 = validation['predictable'].get('ks_annual_pass', [])
                    for i, (h, a) in enumerate(zip(ks_h, ks_1)):
                        st.write(f"Path {i+1}: H={'✓' if h else '✗'} | Ann={'✓' if a else '✗'}")
                else:
                    st.write("No data available")
            
            with tab2:
                col1 = st.columns(1)[0]
                with col1:
                    st.write("**Predictable**")
                    if 'predictable' in validation:
                        bounds = validation['predictable'].get('bounds_pass', [])
                        for i, b in enumerate(bounds):
                            st.write(f"Path {i+1}: {'✓' if b else '✗'}")
                    else:
                        st.write("No data available")
                
            
            with tab3:
                col1 = st.columns(1)[0]
                with col1:
                    st.write("**Predictable**")
                    if 'predictable' in validation:
                        vol = validation['predictable'].get('volatility_pass', [])
                        for i, v in enumerate(vol):
                            st.write(f"Path {i+1}: {'✓' if v else '✗'}")
                    else:
                        st.write("No data available")
                
            with tab4:
                col1 = st.columns(1)[0]
                with col1:
                    st.write("**Predictable**")
                    if 'predictable' in validation:
                        autocorrs = validation['predictable'].get('autocorrelations', [])
                        for i, ac in enumerate(autocorrs):
                            st.write(f"Path {i+1}: {ac:.4f}")
                    else:
                        st.write("No data available")
       
        
        st.divider()
    
    # Individual Path Details Table (PRIORITY 3)
    if validation and ('predictable' in validation):
        st.subheader("📋 Path Details")
        
        # Build dataframe
        path_details = []
        
        # Predictable paths
        if 'predictable' in validation and results['correlations_pred'] is not None:
            val_pred = validation['predictable']
            for i in range(len(results['correlations_pred'])):
                reg = val_pred['regressions'][i]
                path_details.append({
                    'Path ID': i + 1,
                    'Correlation': f"{results['correlations_pred'][i]:.4f}",
                    'β(signal)': f"{reg['beta_signal']:.4f}" if not np.isnan(reg['beta_signal']) else 'NaN',
                    'p(signal)': f"{reg['pval_signal']:.4f}" if not np.isnan(reg['pval_signal']) else 'NaN',
                    'R²(signal)': f"{reg['r2_signal']:.4f}" if not np.isnan(reg['r2_signal']) else 'NaN',
                    'β(lagged return)': f"{reg['beta_lag']:.4f}" if not np.isnan(reg['beta_lag']) else 'NaN',
                    'p(lagged return)': f"{reg['pval_lag']:.4f}" if not np.isnan(reg['pval_lag']) else 'NaN',
                    'R²(lagged return)': f"{reg['r2_lag']:.4f}" if not np.isnan(reg['r2_lag']) else 'NaN',
                    'KS(h-yr)': '✓' if val_pred['ks_h_pass'][i] else '✗',
                    'KS(ann)': '✓' if val_pred['ks_annual_pass'][i] else '✗',
                    'Bounds': '✓' if val_pred['bounds_pass'][i] else '✗',
                    'Vol': '✓' if val_pred['volatility_pass'][i] else '✗',
                    'Status': 'Full ✓' if val_pred['full_compliant'][i] else 'Partial'
                })
        

        
        if path_details:  # Only show table if there are paths
            df_paths = pd.DataFrame(path_details)
            st.dataframe(df_paths, use_container_width=True, height=400)
        
        st.divider()
    
    # Correlation distributions
    if results['correlations_pred'] is not None:
        st.subheader("Correlation Distributions")
        
        fig = go.Figure()
        
        if results['correlations_pred'] is not None:
            fig.add_trace(go.Histogram(
                x=results['correlations_pred'],
                name='Predictable',
                marker_color='#3b82f6',
                opacity=0.7,
                nbinsx=20
            ))
        
    
        fig.update_layout(
            barmode='overlay',
            xaxis_title='Correlation',
            yaxis_title='Count',
            showlegend=True,
            height=400,
            template='plotly_white'
        )
    
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Sample paths visualization with Regression Statistics (PRIORITY 5)
    st.subheader("Sample Paths")
    shared_y_range = compute_shared_y_range(results, config)

    if config.n_paths_predictable > 0 and st.session_state.get('show_sample_paths', False):
        for i in range(min(3, config.n_paths_predictable)):
            fig = build_path_figure(results, config, i, include_realized_return=False)
            apply_path_chart_layout(fig, shared_y_range)
            st.plotly_chart(fig, use_container_width=True)
            fig_realized = build_path_figure(results, config, i, include_realized_return=True)
            apply_path_chart_layout(fig_realized, shared_y_range)
            st.plotly_chart(fig_realized, use_container_width=True)

            # Regression statistics (unchanged)
            if validation and 'predictable' in validation:
                with st.expander(f"📊 Path {i+1} Regression Statistics"):
                    reg = validation['predictable']['regressions'][i]
                    autocorr = validation['predictable']['autocorrelations'][i]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Signal Regression**")
                        status = "✓" if reg['pval_signal'] < 0.05 else "✗"
                        st.write(f"β = {reg['beta_signal']:.4f} {status}")
                        st.write(f"p-value = {reg['pval_signal']:.4f}")
                        st.write(f"R² = {reg['r2_signal']:.4f}")

                    with col2:
                        st.write("**Lag Regression**")
                        status = "✓" if reg['pval_lag'] > 0.05 else "✗"
                        st.write(f"β = {reg['beta_lag']:.4f} {status}")
                        st.write(f"p-value = {reg['pval_lag']:.4f}")
                        st.write(f"R² = {reg['r2_lag']:.4f}")

                    st.write(f"**Autocorrelation:** ρ = {autocorr:.4f}")
    elif config.n_paths_predictable > 0:
        st.caption("Enable `Render Sample Path Charts` in the sidebar to display the interactive path figures.")
    else:
        st.info("No paths generated")

    if config.n_paths_predictable > 0 and st.session_state.get('enable_png_export', False):
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as z:
            for i in range(config.n_paths_predictable):
                try:
                    fig = build_path_figure(results, config, i, include_realized_return=False)
                    apply_path_chart_layout(fig, shared_y_range)

                    png_bytes = fig_to_png_bytes(fig, scale=2)
                    fname = f"predictable_path_{i+1}.png"
                    z.writestr(fname, png_bytes)

                except Exception as e:
                    st.warning(f"PNG export failed for path {i+1}: {e}")


                # === REALIZED RETURNS PNG (41 returns) ===
                try:
                    fig_realized = build_path_figure(results, config, i, include_realized_return=True)
                    apply_path_chart_layout(fig_realized, shared_y_range)

                    png_bytes_realized = fig_to_png_bytes(fig_realized, scale=2)
                    fname_realized = f"predictable_path_{i+1}_r.png"
                    z.writestr(fname_realized, png_bytes_realized)

                except Exception as e:
                    st.warning(f"Realized PNG export failed for path {i+1}: {e}")

        st.download_button(
            label="Download All Plots (PNG ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"all_plots_png_{st.session_state.run_id}.zip",
            mime="application/zip",
            use_container_width=True
        )
    elif config.n_paths_predictable > 0:
        st.caption("Enable `Enable PNG Plot Export` in the sidebar to generate the PNG ZIP download.")

    st.divider()
    
    # Export Controls (PRIORITY 4)
    st.header("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download path details CSV
        if validation  and 'df_paths' in locals():
            csv = df_paths.to_csv(index=False)
            st.download_button(
                label="Download All Data (CSV)",
                data=csv,
                file_name=f"path_details_{st.session_state.run_id}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No path table available to export.")

        # CSV Export of Return Paths (Request id: 4)
        # CSV Export of Return Paths + aligned signal (wide format)
        if st.session_state.get('enable_csv_export', False):

            H = config.time_horizon
            T_is = config.years_insample  # = 40H

            if results['returns_pred'] is not None:

                time_labels = np.arange(-40, 1)
                rows = []

                for i in range(results['returns_pred'].shape[1]):

                    # 41 aligned H-year returns, including the one-period-ahead return at t=0
                    h_returns = results['returns_pred'][H:T_is + H + 1:H, i]

                    # Fitted signal exactly as shown in the graph
                    fitted_signal = results['signal_pred'][0:T_is + H + 1:H, i]

                    row = {"Path_ID": f"path_{i+1}"}

                    # Add signal columns
                    for t, val in zip(time_labels, fitted_signal):
                        row[f"signal_{t}"] = val

                    # Add return columns
                    for t, val in zip(time_labels, h_returns):
                        row[f"return_{t}"] = val

                    rows.append(row)

                df_export = pd.DataFrame(rows)
                csv_export = df_export.to_csv(index=False)

                st.download_button(
                    label="Download Paths + Signals (Wide CSV)",
                    data=csv_export,
                    file_name=f"paths_wide_{st.session_state.run_id}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No paths to export")
        elif 'enable_csv_export' in st.session_state and not st.session_state.enable_csv_export:
             st.caption("Enable CSV export in sidebar to download full path data")

        # HTML Plots Export (Request id: 7)
        if st.session_state.get('enable_plots_export', False):
            # We need to regenerate the figures or store them. 
            # Re-generating them here is cleaner than storing figure objects in session state (memory heavy)
            
            # Create a zip file in memory
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Predictable paths
                if config.n_paths_predictable > 0:
                    for i in range(min(config.n_example_plots // 2, config.n_paths_predictable)):
                        # Re-create figure logic (simplified for export)
                        fig = make_subplots()
                        H = config.time_horizon
                        T_is = config.years_insample
                        
                        r_plot = results['returns_pred'][H:T_is+1:H, i] * 100
                        s_plot = results['signal_pred'][0:T_is+H+1:H, i] * 100

                        x_returns = np.arange(-40, 0)
                        x_signal = np.arange(-40, 1)

                        fig.add_trace(go.Scatter(
                            x=x_returns, y=r_plot,
                            name='Return',
                            line=RETURN_LINE_STYLE
                        ))

                        fig.add_trace(go.Scatter(
                            x=x_signal, y=s_plot,
                            name='Predictive signal',
                            line=SIGNAL_LINE_STYLE
                        ))

                        apply_path_chart_layout(fig, shared_y_range)
                        
                        # Save to HTML string
                        html_str = fig.to_html(full_html=True, include_plotlyjs='cdn')
                        zip_file.writestr(f"predictable_path_{i+1}.html", html_str)
                
            # Download button
            st.download_button(
                label="Download Plots (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"plots_{st.session_state.run_id}.zip",
                mime="application/zip",
                use_container_width=True
            )
        elif 'enable_plots_export' in st.session_state and not st.session_state.enable_plots_export:
             st.caption("Enable Plots export in sidebar to download interactive HTML graphs")
    
    with col2:
        # Download summary JSON
        summary_data = {
            'run_id': st.session_state.run_id,
            'config': {
                'time_horizon': config.time_horizon,
                'target_correlation': config.target_correlation,
                'n_paths_predictable': config.n_paths_predictable,
            },
            'correlations_pred': results['correlations_pred'].tolist() if results['correlations_pred'] is not None else [],
        }
        
        # Convert validation data, handling numpy types
        if validation:
            import copy
            validation_copy = copy.deepcopy(validation)
            
            # Convert numpy bools to Python bools in validation dict
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                else:
                    return obj
            
            summary_data['validation'] = convert_numpy_types(validation_copy)
        
        json_str = json.dumps(summary_data, indent=2)
        st.download_button(
            label="Download Summary (JSON)",
            data=json_str,
            file_name=f"summary_{st.session_state.run_id}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        # Save to disk
        if st.button("💾 Save to Disk", use_container_width=True):
            save_dir = Path(__file__).parent / "saved_runs" / st.session_state.run_id
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save config
            with open(save_dir / "config.json", 'w') as f:
                json.dump(summary_data['config'], f, indent=2)
            
            # Save validation
            with open(save_dir / "validation.json", 'w') as f:
                json.dump(validation, f, indent=2)
            
            # Save paths
            np.savez(
                save_dir / "paths.npz",
                returns_pred=results['returns_pred'],
                signal_pred=results['signal_pred'],
                signal_pred_actual=results['signal_pred_actual'],
                )
            
            st.success(f"✓ Saved to {save_dir}")
    
    
else:
    st.info("👈 Configure parameters in the sidebar and click 'Run Simulation' to get started")
