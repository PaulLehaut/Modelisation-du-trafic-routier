import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

# ==========================================
# 1. PINN Architecture
# ==========================================
class TrafficPINN(nn.Module):
    def __init__(self, t_max, x_max, layers=[2, 32, 32, 32, 32, 1]):
        super().__init__()
        self.t_max = t_max
        self.x_max = x_max
        
        self.hidden_layers = nn.ModuleList()
        for i in range(len(layers) - 2):
            self.hidden_layers.append(nn.Linear(layers[i], layers[i+1]))
            self.hidden_layers.append(nn.Tanh())
            
        self.output_layer = nn.Linear(layers[-2], layers[-1])
        self.sigmoid = nn.Sigmoid()

    def forward(self, t, x):
        # Internal Normalization to [-1, 1]
        t_norm = 2.0 * (t / self.t_max) - 1.0
        x_norm = 2.0 * (x / self.x_max) - 1.0
        
        inputs = torch.cat([t_norm, x_norm], dim=1)
        for layer in self.hidden_layers:
            inputs = layer(inputs)
            
        return self.sigmoid(self.output_layer(inputs))

# ==========================================
# 2. Physics Loss Computation
# ==========================================
def compute_physics_loss(model, t_colloc, x_colloc, v_max, gamma=0.05):
    t_colloc.requires_grad_(True)
    x_colloc.requires_grad_(True)
    
    rho = model(t_colloc, x_colloc)
    
    rho_t = torch.autograd.grad(rho, t_colloc, grad_outputs=torch.ones_like(rho), create_graph=True)[0]
    rho_x = torch.autograd.grad(rho, x_colloc, grad_outputs=torch.ones_like(rho), create_graph=True)[0]
    rho_xx = torch.autograd.grad(rho_x, x_colloc, grad_outputs=torch.ones_like(rho_x), create_graph=True)[0]
    
    # Viscous Greenshields LWR Residual
    flux_derivative = v_max * (1.0 - 2.0 * rho)
    residual = rho_t + flux_derivative * rho_x - gamma * rho_xx
    
    return torch.mean(residual**2)

# ==========================================
# 3. Data Loading & LHS Generation
# ==========================================
def load_and_split_data(csv_path, pv_ratio=0.10, v_max=50.0):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Convert Time_s to Time_h for physical consistency with Velocity_kmh
    df['Time_h'] = df['Time_s'] / 3600.0
    
    # Calculate Ground Truth Density using inverted Greenshields
    df['Density'] = 1.0 - (df['Velocity_kmh'] / v_max)
    
    t_max = df['Time_h'].max()
    x_max = df['Position_km'].max()
    
    # Identify unique vehicles and sample 10% for PVs
    unique_vehicles = df['Vehicle_ID'].unique()
    num_pvs = int(len(unique_vehicles) * pv_ratio)
    
    # Randomly select PVs
    np.random.seed(42) # For reproducibility
    pv_ids = np.random.choice(unique_vehicles, num_pvs, replace=False)
    
    # Split Dataset
    train_df = df[df['Vehicle_ID'].isin(pv_ids)]
    test_df = df[~df['Vehicle_ID'].isin(pv_ids)] # The 90% unobserved
    
    print(f"Data Split: {len(train_df)} training points (10%), {len(test_df)} test points (90%).")
    return train_df, test_df, t_max, x_max, pv_ids

def generate_lhs_collocation(num_points, t_max, x_max):
    # Latin Hypercube Sampling using SciPy
    sampler = qmc.LatinHypercube(d=2, seed=42)
    sample = sampler.random(n=num_points)
    
    # Scale from [0, 1] to physical bounds
    t_colloc = sample[:, 0] * t_max
    x_colloc = sample[:, 1] * x_max
    
    return t_colloc, x_colloc


# ==========================================
# 4. Adaptive Weight Calculation
# ==========================================
def calculate_pv_weights(train_df, pv_ids, max_weight_cap=5.0):
    """
    Calculates weights inversely proportional to the temporal density (mean time gap)
    between adjacent Probe Vehicles.
    """
    # Calculate the average entry time (or general time presence) for each PV
    pv_mean_times = train_df.groupby('Vehicle_ID')['Time_h'].mean().sort_values()
    sorted_pvs = pv_mean_times.index.tolist()
    
    weights_dict = {}
    n_pvs = len(sorted_pvs)
    
    for i, pv in enumerate(sorted_pvs):
        # Calculate temporal gap to previous and next PVs
        gap_prev = pv_mean_times.iloc[i] - pv_mean_times.iloc[i-1] if i > 0 else 0
        gap_next = pv_mean_times.iloc[i+1] - pv_mean_times.iloc[i] if i < n_pvs - 1 else 0
        
        # Edge cases: first and last PVs duplicate their only available gap
        if i == 0: gap_prev = gap_next
        if i == n_pvs - 1: gap_next = gap_prev
            
        mean_gap = (gap_prev + gap_next) / 2.0
        weights_dict[pv] = mean_gap
        
    # Convert to inversely proportional weights: larger gap = higher weight
    # We add a tiny epsilon to avoid division by zero
    epsilon = 1e-6
    raw_weights = np.array([1.0 / (weights_dict[pv] + epsilon) for pv in pv_ids])
    
    # Normalize weights so they average to 1.0 (keeps learning rate stable)
    normalized_weights = raw_weights / np.mean(raw_weights)
    
    # Cap the maximum weight to prevent a single isolated PV from destroying the gradient
    clipped_weights = np.clip(normalized_weights, a_min=0.1, a_max=max_weight_cap)
    
    # Map back to a dictionary
    final_weights = {pv: weight for pv, weight in zip(pv_ids, clipped_weights)}
    
    # Create a column in the dataframe so it easily maps to the PyTorch tensors
    train_df = train_df.copy()
    train_df['Weight'] = train_df['Vehicle_ID'].map(final_weights)
    
    return train_df

# ==========================================
# 5. Training without adaptative weights
# ==========================================
def train_pinn(csv_path, epochs=5000, num_colloc=10000, mu=0.5, gamma=0.05, v_max=50.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # 1. Prepare Data
    train_df, test_df, t_max, x_max, pv_ids = load_and_split_data(csv_path, pv_ratio=0.10, v_max=v_max)
    
    # Convert training data to tensors
    t_data = torch.tensor(train_df['Time_h'].values, dtype=torch.float32).view(-1, 1).to(device)
    x_data = torch.tensor(train_df['Position_km'].values, dtype=torch.float32).view(-1, 1).to(device)
    rho_data = torch.tensor(train_df['Density'].values, dtype=torch.float32).view(-1, 1).to(device)
    
    # Convert test data for Current Estimation Error (CEE)
    t_test = torch.tensor(test_df['Time_h'].values, dtype=torch.float32).view(-1, 1).to(device)
    x_test = torch.tensor(test_df['Position_km'].values, dtype=torch.float32).view(-1, 1).to(device)
    rho_test = torch.tensor(test_df['Density'].values, dtype=torch.float32).view(-1, 1).to(device)
    
    # Generate LHS points and convert to tensors
    t_c_np, x_c_np = generate_lhs_collocation(num_colloc, t_max, x_max)
    t_colloc = torch.tensor(t_c_np, dtype=torch.float32).view(-1, 1).to(device)
    x_colloc = torch.tensor(x_c_np, dtype=torch.float32).view(-1, 1).to(device)
    
    # 2. Initialize Model and Optimizer
    model = TrafficPINN(t_max, x_max).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 3. Training Loop
    epochs_logged = []
    loss_total_history = []
    loss_data_history = []
    loss_phys_history = []
    cee_history = []
    
    # Track the best model based on CEE (Simple Early Stopping mechanic)
    best_cee = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Supervised Data Loss (PVs)
        rho_pred = model(t_data, x_data)
        loss_data = torch.mean((rho_pred - rho_data)**2)
        
        # Unsupervised Physics Loss (LHS Collocation points)
        loss_phys = compute_physics_loss(model, t_colloc, x_colloc, v_max, gamma)
        
        # Combined Loss
        loss_total = mu * loss_data + (1.0 - mu) * loss_phys
        
        loss_total.backward()
        optimizer.step()
        
        # 4. Evaluation and Logging
        if epoch % 100 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                # Compute CEE on the 90% unobserved data
                rho_test_pred = model(t_test, x_test)
                cee = torch.mean((rho_test_pred - rho_test)**2).item()
                
                # Save best model
                if cee < best_cee:
                    best_cee = cee
                    best_model_state = model.state_dict()
                
            epochs_logged.append(epoch)
            loss_total_history.append(loss_total.item())
            loss_data_history.append(loss_data.item())
            loss_phys_history.append(loss_phys.item())
            cee_history.append(cee)
            
            print(f"Epoch {epoch:04d} | Total: {loss_total.item():.6f} | Data: {loss_data.item():.6f} | Phys: {loss_phys.item():.6f} | CEE: {cee:.6f}")
            
    # Restore the best weights before returning
    model.load_state_dict(best_model_state)
    print(f"Training finished. Restored best model weights with CEE: {best_cee:.6f}")
    
    return model, epochs_logged, loss_total_history, loss_data_history, loss_phys_history, cee_history

# ==========================================
# Visualization Function
# ==========================================
def train_pinn_with_weights(csv_path, epochs=5000, num_colloc=10000, mu=0.1, gamma=0.05, v_max=50.0, use_adaptive=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # 1. Prepare Data
    train_df, test_df, t_max, x_max, pv_ids = load_and_split_data(csv_path, pv_ratio=0.10, v_max=v_max)

    # Apply weights if adaptive mode is ON
    if use_adaptive:
        train_df = calculate_pv_weights(train_df, pv_ids)
        w_data = torch.tensor(train_df['Weight'].values, dtype=torch.float32).view(-1, 1).to(device)
    else:
        # Uniform weights of 1.0
        w_data = torch.ones((len(train_df), 1), dtype=torch.float32).to(device)
    
    # Convert training data to tensors
    t_data = torch.tensor(train_df['Time_h'].values, dtype=torch.float32).view(-1, 1).to(device)
    x_data = torch.tensor(train_df['Position_km'].values, dtype=torch.float32).view(-1, 1).to(device)
    rho_data = torch.tensor(train_df['Density'].values, dtype=torch.float32).view(-1, 1).to(device)
    
    # Convert test data for Current Estimation Error (CEE)
    t_test = torch.tensor(test_df['Time_h'].values, dtype=torch.float32).view(-1, 1).to(device)
    x_test = torch.tensor(test_df['Position_km'].values, dtype=torch.float32).view(-1, 1).to(device)
    rho_test = torch.tensor(test_df['Density'].values, dtype=torch.float32).view(-1, 1).to(device)
    
    # Generate LHS points and convert to tensors
    t_c_np, x_c_np = generate_lhs_collocation(num_colloc, t_max, x_max)
    t_colloc = torch.tensor(t_c_np, dtype=torch.float32).view(-1, 1).to(device)
    x_colloc = torch.tensor(x_c_np, dtype=torch.float32).view(-1, 1).to(device)
    
    # 2. Initialize Model and Optimizer
    model = TrafficPINN(t_max, x_max).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 3. Training Loop
    epochs_logged = []
    loss_total_history = []
    loss_data_history = []
    loss_phys_history = []
    cee_history = []
    
    # Track the best model based on CEE (Simple Early Stopping mechanic)
    best_cee = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Supervised Weighing Data Loss (PVs)
        rho_pred = model(t_data, x_data)
        # Multiply the squared error by the personalized PV weights
        loss_data = torch.mean(w_data * (rho_pred - rho_data)**2)
        
        # Unsupervised Physics Loss (LHS Collocation points)
        loss_phys = compute_physics_loss(model, t_colloc, x_colloc, v_max, gamma)
        
        # Combined Loss
        loss_total = mu * loss_data + (1.0 - mu) * loss_phys
        
        loss_total.backward()
        optimizer.step()
        
        # 4. Evaluation and Logging
        if epoch % 100 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                # Compute CEE on the 90% unobserved data
                rho_test_pred = model(t_test, x_test)
                cee = torch.mean((rho_test_pred - rho_test)**2).item()
                
                # Save best model
                if cee < best_cee:
                    best_cee = cee
                    best_model_state = model.state_dict()
                
            epochs_logged.append(epoch)
            loss_total_history.append(loss_total.item())
            loss_data_history.append(loss_data.item())
            loss_phys_history.append(loss_phys.item())
            cee_history.append(cee)
            
            print(f"Epoch {epoch:04d} | Total: {loss_total.item():.6f} | Data: {loss_data.item():.6f} | Phys: {loss_phys.item():.6f} | CEE: {cee:.6f}")
            
    # Restore the best weights before returning
    model.load_state_dict(best_model_state)
    print(f"Training finished. Restored best model weights with CEE: {best_cee:.6f}")
    
    return model, epochs_logged, loss_total_history, loss_data_history, loss_phys_history, cee_history


# ==========================================
# Visualization Unweighted data Function
# ==========================================
def plot_training_metrics(epochs_logged, loss_total, loss_data, loss_phys, cee, mu, epochs):
    """
    Generates a 2-panel plot showing the internal PINN losses and the external CEE validation metric.
    Uses a logarithmic scale on the y-axis since MSE losses can drop exponentially.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Panel 1: PINN Optimization Losses
    ax1.plot(epochs_logged, loss_total, label='Total Loss', color='black', linewidth=2)
    ax1.plot(epochs_logged, loss_data, label='Data Loss (10% PVs)', color='blue', alpha=0.8)
    ax1.plot(epochs_logged, loss_phys, label='Physics Loss (PDE)', color='green', alpha=0.8)
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Mean Squared Error (Log Scale)', fontsize=12)
    ax1.set_title('PINN Internal Training Losses', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    
    # Panel 2: Current Estimation Error (Validation)
    ax2.plot(epochs_logged, cee, label='CEE (90% Hidden Traffic)', color='red', linewidth=2)
    
    # Optional: Plot a point showing the minimum CEE achieved
    min_cee = min(cee)
    min_epoch = epochs_logged[cee.index(min_cee)]
    ax2.scatter(min_epoch, min_cee, color='darkred', zorder=5, 
                label=f'Min CEE: {min_cee:.4f} at epoch {min_epoch}')
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('CEE (Log Scale)', fontsize=12)
    ax2.set_title('Validation: Current Estimation Error', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(f"graphs/pinn_training_metrics_mu_{mu}_epochs_{epochs}.png", dpi=150)
    plt.show()

# ==========================================
# Visualization Weighted and Unweighted data Function
# ==========================================
def plot_comparative_training_metrics(
    epochs_logged, loss_total, loss_data, loss_phys, cee,
    epochs_logged_adapt, loss_total_adapt, loss_data_adapt, loss_phys_adapt, cee_adapt,
    mu, epochs
):
    """
    Generates a 2-panel plot comparing Uniform vs Adaptive weights.
    Panel 1: Internal PINN Losses (Total, Data, Phys)
    Panel 2: Validation Metric (CEE)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Uniform PV Weights vs. Adaptive Inverse-Density Weights', fontsize=16)
    # ---------------------------------------------------------
    # Panel 1: PINN Optimization Losses (Total, Data, Phys)
    # ---------------------------------------------------------
    # Baseline (Uniform) - Solid Lines
    ax1.plot(epochs_logged, loss_total, label='Base Total', color='black', linestyle='-', linewidth=2)
    ax1.plot(epochs_logged, loss_data, label='Base Data (PVs)', color='blue', linestyle='-', alpha=0.7)
    ax1.plot(epochs_logged, loss_phys, label='Base Phys (PDE)', color='green', linestyle='-', alpha=0.7)
    
    # Adaptive - Dashed Lines
    ax1.plot(epochs_logged_adapt, loss_total_adapt, label='Adapt Total', color='black', linestyle='--', linewidth=2)
    ax1.plot(epochs_logged_adapt, loss_data_adapt, label='Adapt Data (PVs)', color='blue', linestyle='--', alpha=0.7)
    ax1.plot(epochs_logged_adapt, loss_phys_adapt, label='Adapt Phys (PDE)', color='green', linestyle='--', alpha=0.7)
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Mean Squared Error (Log Scale)', fontsize=12)
    ax1.set_title('Internal Training Losses', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    
    # ---------------------------------------------------------
    # Panel 2: Current Estimation Error (Validation)
    # ---------------------------------------------------------
    # Baseline (Uniform) - Solid Line
    ax2.plot(epochs_logged, cee, label='Baseline CEE', color='red', linestyle='-', linewidth=2)
    
    # Adaptive - Dashed Line
    ax2.plot(epochs_logged_adapt, cee_adapt, label='Adaptive CEE', color='red', linestyle='--', linewidth=2)
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('CEE (Log Scale)', fontsize=12)
    ax2.set_title('Validation: Current Estimation Error', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(f"graphs/pinn_comparative_metrics_mu_{mu}_epochs_{epochs}.png", dpi=150)
    plt.show()


# ==========================================
# Run the Code
# ==========================================
if __name__ == '__main__':
    # Make sure this points to one of the CSVs you generated
    csv_file = "datasets/traffic_N1000_shock.csv" 

    WEIGHT = True
    
    mu = 0.9
    epochs = 200
    num_colloc = 10000
    gamma = 0.05

    if not WEIGHT:
        if os.path.exists(csv_file):
            trained_model, trained_epochs, loss_total, loss_data, loss_phys, cees = train_pinn(
                csv_file, 
                epochs, 
                num_colloc,   # Number of LHS points
                mu,           # Weighting parameter
                gamma         # Diffusion parameter
            )
            plot_training_metrics(epochs, loss_total, loss_data, loss_phys, cees, mu, epochs)
        else:
            print(f"Dataset {csv_file} not found. Ensure the generation script was run first.")
    
    else:
        if os.path.exists(csv_file):
            trained_model, trained_epochs, loss_total, loss_data, loss_phys, cees = train_pinn_with_weights(
                csv_file, 
                epochs, 
                num_colloc,   # Number of LHS points
                mu,           # Weighting parameter
                gamma,        # Diffusion parameter
                use_adaptive = False
            )
            trained_model_adapt, trained_epochs_adapt, loss_total_adapt, loss_data_adapt, loss_phys_adapt, cees_adapt = train_pinn_with_weights(
                csv_file, 
                epochs, 
                num_colloc,   # Number of LHS points
                mu,           # Weighting parameter
                gamma,        # Diffusion parameter
                use_adaptive = True
            )
            plot_comparative_training_metrics(
                trained_epochs, loss_total, loss_data, loss_phys, cees,
                trained_epochs_adapt, loss_total_adapt, loss_data_adapt, loss_phys_adapt, cees_adapt,
                mu, epochs
            )
        else:
            print(f"Dataset {csv_file} not found. Ensure the generation script was run first.")


