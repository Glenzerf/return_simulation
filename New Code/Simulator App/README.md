# Path Simulation App

A clean, interactive Streamlit application for running predictable return path simulations with optional validation against S&P 500 historical data.

## Features

- **Interactive Simulation**: Run Monte Carlo simulations of predictable return paths with customizable parameters.
- **Validation**: Compare simulated paths against actual S&P 500 return distributions (5-year and annual).
- **Visualization**: View correlation statistics and path details.
- **Presets**: Quickly switch between standard configurations (e.g., 2-year vs 5-year horizons).

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1.  **Navigate to the app directory**:
    ```bash
    cd "New Code/Simulator App"
    ```

2.  **Create a virtual environment (recommended)**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On macOS/Linux
    # .venv\Scripts\activate   # On Windows
    ```

3.  **Install dependencies**:
    ```bash
    python -m pip install -r requirements.txt
    ```

## Usage

1.  **Start the application**:
    ```bash
    streamlit run app.py
    ```

2.  **Open your browser**:
    The app should automatically open in your default browser at `http://localhost:8501`.

3.  **Run a simulation**:
    - Select a **Preset** (e.g., "Current (2-year)") or choose "Custom".
    - Adjust parameters if needed (Time Horizon, Target Correlation, Predictable Paths, etc.).
    - Ensure **Enable SPX Validation** is checked if you have the `SPX_5Y_Returns.xlsx` data file (autodetected if in standard location).
    - Click **🚀 Run Simulation**.

## Project Structure

- `app.py`: Main Streamlit application and UI logic.
- `simulation_engine.py`: Core logic for generating predictable paths and performing validation.
- `requirements.txt`: Python package dependencies.
