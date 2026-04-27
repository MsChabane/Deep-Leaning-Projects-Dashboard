import streamlit as st
import json
from PIL import Image
from graphviz import Digraph
import os

st.set_page_config(page_title="Deep Learning Project Dashboard", layout="wide", )

st.title("📊 Deep Learning Projects Dashboard")

# --- Helper Function for Architecture Visualization ---
def render_architecture_graph(architecture_data, title="Architecture"):
    """
    Renders a Graphviz Digraph from a list of layer dictionaries.
    Handles Linear, LSTM, and Activation layers.
    """
    dot = Digraph(comment=title)
    dot.attr(rankdir='LR', splines='ortho')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='11')
    dot.attr('edge', penwidth='2')

    # Define a color scheme for different layer types for consistency
    COLORS = {
        "Linear": "#90CAF9",   # Light Blue
        "LSTM": "#FFCC80",     # Light Orange (for RNNs)
        "ReLU": "#A5D6A7",     # Light Green (for Activations)
        "Sigmoid": "#A5D6A7",  # Light Green
        "Tanh": "#A5D6A7",     # Light Green
        "Conv2d": "#FFEB3B",   # Yellow (if you add CNNs later)
        "Dropout": "#B0BEC5",  # Greyish
        "default": "#E0E0E0"   # Very Light Grey
    }

    prev_node = None
    for i, layer in enumerate(architecture_data):
        # Use .get() for safer access to dictionary keys, providing a default if not found
        layer_type = layer.get("type", "Unknown")
        node_id = layer.get("name", f"layer_{i}") # Fallback node ID
        label = layer_type
        fillcolor = COLORS.get(layer_type, COLORS["default"])

        if layer_type == "Linear":
            in_f = layer.get("in_features", "?")
            out_f = layer.get("out_features", "?")
            label = f"Linear\n{in_f} → {out_f}"
        elif layer_type == "LSTM":
            # Assuming 'input_size' from your get_model_architecture for LSTM
            # If your JSON for LSTM saves it as 'in_features', adjust here.
            input_s = layer.get("input_size", "?") 
            hidden_s = layer.get("hidden_size", "?")
            num_l = layer.get("num_layers", "?")
            drp = layer.get("dropout", "?")
            label = (f"LSTM\n"
                     f"Input: {input_s} | Hidden: {hidden_s}\n"
                     f"Layers: {num_l} | Dropout: {drp}")
        elif layer_type in ["ReLU", "Sigmoid", "Tanh", "GELU", "LeakyReLU"]:
            # For activation layers, just show their type
            label = layer_type
        # Add more layer types if needed (e.g., Conv2d, BatchNorm, etc.)

        dot.node(node_id, label, fillcolor=fillcolor)

        if prev_node:
            dot.edge(prev_node, node_id)
        prev_node = node_id
    
    return dot

# --- Data Loading Helper ---
@st.cache_data
def load_json_data(filepath):
    """Loads JSON data from a file, handles errors."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from {filepath}")
        return None

@st.cache_data
def load_image_data(filepath):
    """Loads image data from a file, handles errors."""
    try:
        return Image.open(filepath)
    except FileNotFoundError:
        st.error(f"Error: Image not found at {filepath}")
        return None
    except Exception as e:
        st.error(f"Error loading image {filepath}: {e}")
        return None

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Regression", "🧠 Classification", "📉 Forecasting", "🕵️ Anomaly Detection"])

# --- Tab 1: Regression ---
with tab1:
    st.header("Regression Model")
    
    reg_params = load_json_data("best_reg/params.json")
    reg_architecture = load_json_data("best_reg/architecture.json")

    if reg_params and reg_architecture:
        st.subheader("🗂 Dataset Overview")
        col1, col2 = st.columns(2)
        col1.metric("Name", "Temperature Prediction")
        col1.metric("Target Variable", "Temperature (C)")
        col2.metric("Samples", 96453)
        col2.metric("Features", 12)
        
        st.divider()

        st.subheader("⚙️ Model Parameters")
        col1, col2, col3 = st.columns(3)
        col1.metric("Epochs", reg_params.get("epochs", "N/A"))
        col1.metric("Batch Size", reg_params.get("batch_size", "N/A"))
        col2.metric("Learning Rate", reg_params.get("lr", "N/A"))
        col2.metric("Optimizer", reg_params.get("optimizer", "N/A"))
        col3.metric("Loss Function", reg_params.get("criterion", "N/A"))

        st.divider()

        st.subheader("🏗️ Model Architecture")
        # Regression architecture should be a simple list of layers
        if isinstance(reg_architecture, list):
            st.graphviz_chart(render_architecture_graph(reg_architecture, "Regression Architecture"))
        else:
            st.warning("Regression architecture format not recognized (expected a list of layers).")


        st.divider()
        st.subheader("📊 Performance")
        reg_img = load_image_data("best_reg/reg.png")
        if reg_img:
            st.image(reg_img, caption="Regression Model Training & Validation Metrics")

# --- Tab 2: Classification ---
with tab2:
    st.header("🧠 Classification Model")
    
    clf_params = load_json_data("best_clf/params.json")
    clf_architecture = load_json_data("best_clf/architecture.json")

    if clf_params and clf_architecture:
        st.subheader("🗂 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Name", "Brain Tumor")
        col1.metric("Target Variable", "Tumor")
        col2.metric("Samples", 20000)
        col2.metric("Features", 17)
        col3.metric("Imbalanced", "No")
        
        st.divider()

        st.subheader("⚙️ Model Parameters")
        col1, col2, col3 = st.columns(3)
        col1.metric("Epochs", clf_params.get("epochs", "N/A"))
        col1.metric("Batch Size", clf_params.get("batch_size", "N/A"))
        col2.metric("Learning Rate", clf_params.get("lr", "N/A"))
        col2.metric("Optimizer", clf_params.get("optimizer", "N/A"))
        col3.metric("Loss Function", clf_params.get("criterion", "N/A"))
        col3.metric("Last Layer Activation", str(clf_params.get("with activation in last layer", "N/A")))

        st.divider()

        st.subheader("🏗️ Model Architecture")
        if isinstance(clf_architecture, list):
            st.graphviz_chart(render_architecture_graph(clf_architecture, "Classification Architecture"))
        else:
            st.warning("Classification architecture format not recognized (expected a list of layers).")

        st.divider()
        st.subheader("📊 Performance")
        clf_img = load_image_data("best_clf/B32.png")
        if clf_img:
            st.image(clf_img, caption="Classification Model Training & Validation Metrics")

# --- Tab 3: Forecasting ---
with tab3:
    st.header("Forecasting Model")
    
    frc_params = load_json_data("best_frc/params.json")
    frc_architecture = load_json_data("best_frc/architecture.json")

    if frc_params and frc_architecture:
        st.subheader("🗂 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Name", "Wind Power Forecasting")
        col1.metric("Target Variable", "LV ActivePower (kW)")
        col2.metric("Samples", 50530)
        col2.metric("Features", 5)
        col3.metric("Type", "Univariate")
        col3.metric("Period", "10 min")
        
        st.divider()

        st.subheader("⚙️ Model Parameters")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Epochs", frc_params.get("epochs", "N/A"))
        col1.metric("Batch Size", frc_params.get("batch_size", "N/A"))
        col2.metric("Learning Rate", frc_params.get("lr", "N/A"))
        col2.metric("Optimizer", frc_params.get("optimizer", "N/A"))
        col3.metric("Loss Function", frc_params.get("criterion", "N/A"))
        col4.metric("Window Size", frc_params.get("window size", "N/A"))
        col4.metric("Step", frc_params.get("step", "N/A"))

        st.divider()

        st.subheader("🏗️ Model Architecture")
        if isinstance(frc_architecture, list):
            st.graphviz_chart(render_architecture_graph(frc_architecture, "Forecasting Architecture"))
        else:
            st.warning("Forecasting architecture format not recognized (expected a list of layers).")

        st.divider()
        st.subheader("📊 Performance")
        frc_img = load_image_data("best_frc/image.png")
        if frc_img:
            st.image(frc_img, caption="Forecasting Model Training & Validation Metrics")

# --- Tab 4: Anomaly Detection ---
with tab4:
    st.header("🕵️ Anomaly Detection Model")
    
    anml_params = load_json_data("best_anml/params.json")
    anml_architecture = load_json_data("best_anml/architecture.json")
    anml_results = load_json_data("best_anml/results.json")

    if anml_params and anml_architecture and anml_results:
        st.subheader("🗂 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Name", "Credit Card Fraud")
        col1.metric("Target Variable", "Class (Fraud/Normal)")
        col2.metric("Samples", 284807)
        col2.metric("Features", 31)
        col3.metric("Fraud Ratio", f"{anml_params.get('fraud_ratio', 0.001727):.5f}") # Use a default or load from params

        st.divider()

        st.subheader("⚙️ Model Parameters")
        col1, col2, col3 = st.columns(3)
        col1.metric("Epochs", anml_params.get("epochs", "N/A"))
        col1.metric("Batch Size", anml_params.get("batch_size", "N/A"))
        col2.metric("Learning Rate", anml_params.get("lr", "N/A"))
        col2.metric("Optimizer", anml_params.get("optimizer", "N/A"))
        col3.metric("Loss Function", anml_params.get("criterion", "N/A"))
        col3.metric("Latent Dimension", anml_params.get("Latent space", "N/A"))

        st.divider()

        st.subheader("🏗️ Model Architecture (Autoencoder)")
        # Anomaly detection architecture is typically an Autoencoder with encoder/decoder
        if isinstance(anml_architecture, dict) and "encoder" in anml_architecture and "decoder" in anml_architecture:
            col_enc, col_dec = st.columns(2)
            with col_enc:
                st.write("#### Encoder")
                st.graphviz_chart(render_architecture_graph(anml_architecture["encoder"], "Encoder"))
            with col_dec:
                st.write("#### Decoder")
                st.graphviz_chart(render_architecture_graph(anml_architecture["decoder"], "Decoder"))
        else:
            st.warning("Anomaly Detection architecture format not recognized (expected dict with 'encoder'/'decoder').")

        st.divider()
        
        st.subheader("📊 Performance")
        anml_loss_img = load_image_data("best_anml/LossAnomaly.png")
        if anml_loss_img:
            st.image(anml_loss_img, caption="Anomaly Detection Training & Validation Loss")
        
        st.subheader("📈 Anomaly Detection Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("AUC", f"{anml_results.get('auc', 0.0):.3f}")
        col1.metric("Precision", f"{anml_results.get('precision', 0.0):.3f}")
        col2.metric("Recall", f"{anml_results.get('recall', 0.0):.3f}")
        col2.metric("F1 Score", f"{anml_results.get('f1', 0.0):.3f}")
        col3.metric("Optimal Threshold", f"{anml_results.get('threshold', 0.0):.3f}")
        col3.metric("Quantile Used", f"{anml_results.get('quantile', 'N/A')}") # Corrected typo

        st.info(f"💡 **Anomaly Detection Logic:** An observation is considered an anomaly if its reconstruction error "
                f"is greater than the determined threshold of **{anml_results.get('threshold', 0.0):.3f}**.")
    else:
        st.warning("Could not load all necessary data for Anomaly Detection tab.")

# --- End of Streamlit App ---
