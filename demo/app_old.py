"""Streamlit demo: fly drone with distilled student policy.

This interactive demo allows users to:
1. Load and visualize a trained drone navigation policy
2. Configure simulation parameters (obstacles, weather)
3. View real-time flight metrics (TTC, path deviation)
4. Compare different models (transformer, LSTM, PID)
5. Visualize attention patterns and state predictions
"""
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import streamlit as st
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import imageio
import io

# Import local modules
try:
    import flycraft_gym  # noqa: F401
except ImportError:
    st.error("FlyCraft gym not installed. Install with: pip install flycraft")
    st.stop()

from drone_rl.models.transformer_policy import TransformerActorCritic
from drone_rl.models.baselines import SimpleLSTMPolicy, DronePositionController
from drone_rl.utils.metrics import time_to_collision, path_deviation, velocity_error
from stable_baselines3 import PPO

# Page configuration
st.set_page_config(
    page_title="Drone Transformer RL Demo",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        color: #555;
    }
    .success {
        color: #28a745;
    }
    .warning {
        color: #ffc107;
    }
    .danger {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading to avoid reloading on every interaction
@st.cache_resource
def load_model(model_path: str, model_type: str = "transformer"):
    """Load a trained model from checkpoint.
    
    Parameters
    ----------
    model_path : str
        Path to model checkpoint
    model_type : str
        Type of model (transformer, lstm, pid)
        
    Returns
    -------
    model
        Loaded model
    env
        Environment instance
    """
    try:
        # Create environment
        env = gym.make("FlyCraft-Nav-v0", render_mode="rgb_array", max_episode_steps=1000)
        
        # For PID controller, return a custom controller
        if model_type == "pid":
            controller = DronePositionController()
            return controller, env
        
        # For transformer or LSTM, load from checkpoint
        if model_type == "transformer":
            custom_objects = {"policy_class": TransformerActorCritic}
        elif model_type == "lstm":
            custom_objects = {"policy_class": SimpleLSTMPolicy}
        else:
            custom_objects = {}
            
        model = PPO.load(model_path, env=env, device="cpu", custom_objects=custom_objects)
        return model, env
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def create_trajectory_plot(positions: np.ndarray, reference: Optional[np.ndarray] = None) -> Figure:
    """Create 3D trajectory visualization.
    
    Parameters
    ----------
    positions : np.ndarray
        Array of drone positions [N, 3]
    reference : Optional[np.ndarray]
        Optional reference trajectory [M, 3]
        
    Returns
    -------
    Figure
        Plotly figure with trajectory
    """
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter3d"}]])
    
    # Add drone trajectory
    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="lines+markers",
            name="Drone Path",
            line=dict(color="blue", width=4),
            marker=dict(size=3, color="blue"),
        )
    )
    
    # Add reference trajectory if provided
    if reference is not None:
        fig.add_trace(
            go.Scatter3d(
                x=reference[:, 0],
                y=reference[:, 1],
                z=reference[:, 2],
                mode="lines",
                name="Reference Path",
                line=dict(color="green", width=2, dash="dash"),
            )
        )
    
    # Add start and end points
    fig.add_trace(
        go.Scatter3d(
            x=[positions[0, 0]],
            y=[positions[0, 1]],
            z=[positions[0, 2]],
            mode="markers",
            name="Start",
            marker=dict(size=8, color="green"),
        )
    )
    
    fig.add_trace(
        go.Scatter3d(
            x=[positions[-1, 0]],
            y=[positions[-1, 1]],
            z=[positions[-1, 2]],
            mode="markers",
            name="End",
            marker=dict(size=8, color="red"),
        )
    )
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            zaxis_title="Z Position (m)",
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0, y=1),
        height=500,
    )
    
    return fig

def visualize_attention(attention_weights: np.ndarray, step: int) -> Figure:
    """Visualize attention weights.
    
    Parameters
    ----------
    attention_weights : np.ndarray
        Attention weights [heads, seq_len, seq_len]
    step : int
        Current step to highlight
        
    Returns
    -------
    Figure
        Matplotlib figure with attention visualization
    """
    n_heads = attention_weights.shape[0]
    fig, axes = plt.subplots(1, n_heads, figsize=(n_heads * 3, 3))
    
    if n_heads == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        im = ax.imshow(attention_weights[i], cmap="viridis")
        ax.set_title(f"Head {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Highlight current timestep
        if step < attention_weights.shape[1]:
            ax.axvline(x=step, color="red", linestyle="--", alpha=0.7)
            ax.axhline(y=step, color="red", linestyle="--", alpha=0.7)
    
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.tight_layout()
    
    return fig

def display_metrics(metrics: Dict[str, float]) -> None:
    """Display metrics in styled cards.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of metric names and values
    """
    cols = st.columns(len(metrics))
    
    for i, (name, value) in enumerate(metrics.items()):
        with cols[i]:
            # Determine status color based on metric thresholds
            status = "success"
            if name == "TTC (s)" and value < 3.0:
                status = "danger"
            elif name == "Path Dev (m)" and value > 0.5:
                status = "warning"
            elif name == "Vel Error (%)" and value > 15.0:
                status = "warning"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value {status}">{value:.2f}</div>
                <div class="metric-label">{name}</div>
            </div>
            """, unsafe_allow_html=True)

def run_simulation(model, env, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run simulation with given model and configuration.
    
    Parameters
    ----------
    model : Any
        Model or controller
    env : gym.Env
        Environment instance
    config : Dict[str, Any]
        Simulation configuration
        
    Returns
    -------
    Dict[str, Any]
        Simulation results
    """
    # Reset environment with configured parameters
    obs, info = env.reset(
        options={
            "terrain_id": config.get("terrain_id", 0),
            "weather": {
                "wind_speed": config.get("wind_speed", 0.0),
                "wind_dir": config.get("wind_dir", 0.0),
                "turbulence": config.get("turbulence", 0.0),
            },
            "obstacles": config.get("obstacles", "medium"),
        }
    )
    
    # Prepare result storage
    results = {
        "frames": [],
        "positions": [],
        "velocities": [],
        "actions": [],
        "rewards": [],
        "ttc": [],
        "path_dev": [],
        "vel_error": [],
        "attention": [] if hasattr(model, "policy") and hasattr(model.policy, "get_attention_weights") else None,
    }
    
    # Get reference trajectory if available
    reference_trajectory = info.get("reference_trajectory", None)
    
    # Run simulation
    done = False
    total_reward = 0.0
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    max_steps = config.get("max_steps", 500)
    for step in range(max_steps):
        # Get action from model
        if isinstance(model, DronePositionController):
            # For PID controller
            target_pos = info.get("target_position", np.zeros(3))
            current_pos = info.get("drone_position", np.zeros(3))
            current_vel = info.get("drone_velocity", np.zeros(3))
            current_time = step * 0.05  # Assuming 20Hz simulation
            
            action = model(
                target_pos, current_pos, current_vel, current_time,
                target_yaw=info.get("target_yaw", 0.0),
                current_yaw=info.get("drone_yaw", 0.0)
            )
        else:
            # For RL policies
            action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Update results
        if "rgb_array" in info:
            results["frames"].append(info["rgb_array"])
        
        results["positions"].append(info.get("drone_position", np.zeros(3)))
        results["velocities"].append(info.get("drone_velocity", np.zeros(3)))
        results["actions"].append(action)
        results["rewards"].append(reward)
        
        # Compute metrics
        if "obstacles" in info and len(info["obstacles"]) > 0:
            ttc_val, _ = time_to_collision(
                info["drone_position"],
                info["drone_velocity"],
                info["obstacles"][0][0],  # First obstacle position
                info["obstacles"][0][1],  # First obstacle radius
            )
            results["ttc"].append(ttc_val)
        else:
            results["ttc"].append(float("inf"))
        
        if reference_trajectory is not None:
            dev, _ = path_deviation(
                np.array(results["positions"]),
                reference_trajectory[:len(results["positions"])]
            )
            results["path_dev"].append(dev)
        else:
            results["path_dev"].append(0.0)
        
        if "target_velocity" in info:
            vel_err = velocity_error(
                info["drone_velocity"],
                info["target_velocity"],
                relative=True
            ) * 100.0  # Convert to percentage
            results["vel_error"].append(vel_err)
        else:
            results["vel_error"].append(0.0)
        
        # Capture attention weights if available
        if results["attention"] is not None and hasattr(model.policy, "get_attention_weights"):
            try:
                attn = model.policy.get_attention_weights(obs)
                results["attention"].append(attn)
            except:
                # If attention extraction fails, disable it
                results["attention"] = None
        
        # Update progress
        progress = (step + 1) / max_steps
        progress_bar.progress(progress)
        status_text.text(f"Step {step+1}/{max_steps} | Reward: {reward:.2f} | Total: {total_reward:.2f}")
        
        total_reward += reward
        done = terminated or truncated
        if done:
            break
    
    # Convert lists to arrays
    results["positions"] = np.array(results["positions"])
    results["velocities"] = np.array(results["velocities"])
    results["actions"] = np.array(results["actions"])
    results["rewards"] = np.array(results["rewards"])
    results["ttc"] = np.array(results["ttc"])
    results["path_dev"] = np.array(results["path_dev"])
    results["vel_error"] = np.array(results["vel_error"])
    
    # Add summary metrics
    results["total_reward"] = total_reward
    results["success"] = info.get("success", False)
    results["steps"] = step + 1
    results["reference_trajectory"] = reference_trajectory
    
    # Clear progress bar
    progress_bar.empty()
    status_text.empty()
    
    return results

# Main application
def main():
    """Main Streamlit application."""
    st.title("🚁 Drone Transformer RL – Live Demo")
    st.markdown("""
    This demo showcases our transformer-based reinforcement learning approach for autonomous drone navigation.
    The model learns to navigate through complex environments, avoiding obstacles while following a reference path.
    """)
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["transformer", "lstm", "pid"],
        index=0,
        help="Select model architecture"
    )
    
    # Model checkpoint
    if model_type != "pid":
        checkpoint = st.sidebar.text_input(
            "Model Checkpoint",
            "runs/student_distilled/final_model.zip" if model_type == "transformer" else "runs/baseline_lstm/final_model.zip",
            help="Path to model checkpoint"
        )
    else:
        checkpoint = None
    
    # Environment configuration
    st.sidebar.subheader("Environment")
    
    terrain_id = st.sidebar.selectbox(
        "Terrain",
        [0, 1, 2, 3, 4],
        index=0,
        help="Select terrain type"
    )
    
    wind_speed = st.sidebar.slider(
        "Wind Speed (m/s)",
        0.0, 8.0, 2.0,
        help="Wind speed in meters per second"
    )
    
    wind_dir = st.sidebar.slider(
        "Wind Direction (deg)",
        0.0, 360.0, 45.0,
        help="Wind direction in degrees"
    )
    
    turbulence = st.sidebar.slider(
        "Turbulence",
        0.0, 1.0, 0.2,
        help="Turbulence intensity (0-1)"
    )
    
    obstacles = st.sidebar.selectbox(
        "Obstacles",
        ["none", "easy", "medium", "hard"],
        index=2,
        help="Obstacle density and complexity"
    )
    
    max_steps = st.sidebar.slider(
        "Max Steps",
        100, 1000, 500,
        help="Maximum simulation steps"
    )
    
    # Create configuration dictionary
    config = {
        "terrain_id": terrain_id,
        "wind_speed": wind_speed,
        "wind_dir": wind_dir,
        "turbulence": turbulence,
        "obstacles": obstacles,
        "max_steps": max_steps,
    }
    
    # Load model button
    load_btn = st.sidebar.button("Load Model")
    
    if load_btn:
        with st.spinner(f"Loading {model_type.upper()} model..."):
            model, env = load_model(checkpoint, model_type)
            
        if model is not None and env is not None:
            st.session_state.model = model
            st.session_state.env = env
            st.success(f"{model_type.capitalize()} model loaded successfully!")
            
            # Display model info
            if model_type != "pid":
                st.subheader("Model Information")
                
                # Get parameter count
                if hasattr(model, "policy"):
                    param_count = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
                    st.info(f"Trainable parameters: {param_count:,}")
                    
                    # Display architecture details
                    if model_type == "transformer":
                        if hasattr(model.policy, "features_extractor"):
                            extractor = model.policy.features_extractor
                            if hasattr(extractor, "transformer"):
                                transformer = extractor.transformer
                                st.info(f"""
                                Transformer architecture:
                                - Embedding dimension: {extractor.embed_dim}
                                - Attention heads: {transformer.global_layers[0].num_heads if hasattr(transformer, 'global_layers') else 'N/A'}
                                - Layers: {len(transformer.global_layers) + len(transformer.local_layers) if hasattr(transformer, 'global_layers') else 'N/A'}
                                - Memory: {'Yes' if extractor.use_memory else 'No'}
                                """)
    
    # Run simulation button
    run_btn = st.sidebar.button("Run Simulation")
    
    # Main content
    if "model" in st.session_state and "env" in st.session_state and run_btn:
        model = st.session_state.model
        env = st.session_state.env
        
        # Run simulation
        with st.spinner("Running simulation..."):
            results = run_simulation(model, env, config)
        
        # Display results
        st.subheader("Simulation Results")
        
        # Display success/failure
        if results["success"]:
            st.success(f"Mission successful! Completed in {results['steps']} steps with reward {results['total_reward']:.2f}")
        else:
            st.error(f"Mission failed. Completed {results['steps']} steps with reward {results['total_reward']:.2f}")
        
        # Display metrics
        st.subheader("Flight Metrics")
        metrics = {
            "TTC (s)": np.mean(results["ttc"]) if np.isfinite(results["ttc"]).any() else float("inf"),
            "Path Dev (m)": np.mean(results["path_dev"]),
            "Vel Error (%)": np.mean(results["vel_error"]),
            "Avg Reward": np.mean(results["rewards"]),
        }
        display_metrics(metrics)
        
        # Create columns for visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Display trajectory plot
            st.subheader("3D Trajectory")
            trajectory_fig = create_trajectory_plot(
                results["positions"],
                results["reference_trajectory"]
            )
            st.plotly_chart(trajectory_fig, use_container_width=True)
        
        with col2:
            # Display metrics over time
            st.subheader("Metrics Over Time")
            metrics_df = pd.DataFrame({
                "Step": np.arange(len(results["ttc"])),
                "TTC (s)": np.clip(results["ttc"], 0, 10),  # Cap for visualization
                "Path Deviation (m)": results["path_dev"],
                "Velocity Error (%)": results["vel_error"],
                "Reward": results["rewards"],
            })
            
            # Create multi-line chart
            metrics_fig = go.Figure()
            
            metrics_fig.add_trace(go.Scatter(
                x=metrics_df["Step"], y=metrics_df["TTC (s)"],
                mode="lines", name="TTC (s)", line=dict(color="green")
            ))
            
            metrics_fig.add_trace(go.Scatter(
                x=metrics_df["Step"], y=metrics_df["Path Deviation (m)"],
                mode="lines", name="Path Dev (m)", line=dict(color="blue")
            ))
            
            metrics_fig.add_trace(go.Scatter(
                x=metrics_df["Step"], y=metrics_df["Velocity Error (%)"],
                mode="lines", name="Vel Error (%)", line=dict(color="orange")
            ))
            
            metrics_fig.add_trace(go.Scatter(
                x=metrics_df["Step"], y=metrics_df["Reward"],
                mode="lines", name="Reward", line=dict(color="purple")
            ))
            
            metrics_fig.update_layout(
                xaxis_title="Simulation Step",
                yaxis_title="Value",
                legend=dict(x=0, y=1, orientation="h"),
                margin=dict(l=0, r=0, b=0, t=30),
                height=400,
            )
            
            st.plotly_chart(metrics_fig, use_container_width=True)
        
        # Display video if frames available
        if results["frames"]:
            st.subheader("Flight Video")
            # Convert frames to MP4 in memory
            video_bytes = io.BytesIO()
            imageio.mimsave(video_bytes, results["frames"], format="mp4", fps=20)
            video_bytes.seek(0)
            st.video(video_bytes, format="video/mp4", caption="Flight Animation (MP4)")
        
        # Display attention visualization if available
        if results["attention"] is not None and len(results["attention"]) > 0:
            st.subheader("Attention Visualization")
            
            # Select step for attention visualization
            step_slider = st.slider(
                "Select timestep",
                0, len(results["attention"]) - 1, 0
            )
            
            # Visualize attention for selected step
            attn_fig = visualize_attention(results["attention"][step_slider], step_slider)
            st.pyplot(attn_fig)
            
            st.info("""
            Attention visualization shows how the transformer model focuses on different parts of the input sequence.
            Brighter colors indicate stronger attention. The red lines highlight the current timestep.
            """)
    
    # Display instructions if no model loaded
    if "model" not in st.session_state:
        st.info("👈 Configure and load a model using the sidebar to begin.")
        
        # Display project information
        st.subheader("About the Project")
        st.markdown("""
        ### Drone Transformer RL
        
        This project implements a transformer-based reinforcement learning approach for autonomous drone navigation.
        The system combines:
        
        1. **Hierarchical Transformer Policy** - multi-scale attention with relative position encodings and memory
        2. **Reinforcement Learning** - PPO algorithm for policy optimization
        3. **Knowledge Distillation** - large teacher → lightweight student for real-time inference
        4. **Curriculum Learning** - progressive difficulty for robust performance
        
        The model learns to navigate through complex environments, avoiding obstacles while maintaining desired
        trajectories and velocities.
        
        **Key metrics:**
        - Time-to-Collision (TTC) > 3s (safety)
        - Path Deviation < 0.5m (accuracy)
        - Inference Latency < 10ms (real-time capability)
        """)

if __name__ == "__main__":
    main()
