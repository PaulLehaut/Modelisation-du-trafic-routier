import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import seaborn as sns
from scipy import stats

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
def load_and_split_data(csv_path, pv_ratio=0.10, v_max=50.0, seed = 42):
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
    np.random.seed(seed) # For reproducibility
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

def generate_all_collocation_epochs(num_points, num_epochs, t_max, x_max, device, seed=42):
    print(f"Pre-computing LHS collocation points for {num_epochs} epochs ({num_points} points/epoch)...")
    
    # Generate all M * N points in one single highly-efficient LHS call
    total_points = num_points * num_epochs
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    sample = sampler.random(n=total_points)
    
    # Scale bounds
    t_colloc_np = sample[:, 0] * t_max
    x_colloc_np = sample[:, 1] * x_max
    
    # Convert to tensors and reshape to (num_epochs, num_points, 1)
    t_colloc_tensor = torch.tensor(t_colloc_np, dtype=torch.float32).view(num_epochs, num_points, 1).to(device)
    x_colloc_tensor = torch.tensor(x_colloc_np, dtype=torch.float32).view(num_epochs, num_points, 1).to(device)
    
    return t_colloc_tensor, x_colloc_tensor


# ==========================================
# 4. Adaptive Weight Calculation
# ==========================================

# Proportionnal to spatial density. Problem: give less weight to shockwave so worse CEE
def space_calculate_pv_weights(train_df, max_weight_cap=20.0):
    """
    Calculates weights directly proportional to the mean spatial gap (in km) 
    around each Probe Vehicle averaged over the entire time horizon.
    """
    # 1. Sort by time and position to order vehicles front-to-back at every single second
    df = train_df.sort_values(by=['Time_s', 'Position_km']).copy()
    
    # 2. Calculate the spatial gap (in km) to the vehicle behind and ahead
    # .diff() looks at the previous row (vehicle behind)
    # .diff(-1) looks at the next row (vehicle ahead)
    df['gap_behind'] = df.groupby('Time_s')['Position_km'].diff()
    df['gap_front'] = df.groupby('Time_s')['Position_km'].diff(-1).abs()
    
    # 3. Handle edge cases (first and last PVs at each second)
    df['gap_behind'] = df['gap_behind'].fillna(df['gap_front'])
    df['gap_front'] = df['gap_front'].fillna(df['gap_behind'])
    
    # 4. Mean spatial gap around the vehicle at this exact second
    df['mean_gap_at_t'] = (df['gap_front'] + df['gap_behind']) / 2.0
    
    # 5. Average this gap over the entire 10 minutes for each specific Vehicle_ID
    pv_mean_gaps = df.groupby('Vehicle_ID')['mean_gap_at_t'].mean()
    
    # 6. Assign raw weights (Directly proportional: larger gap = higher weight)
    raw_weights = pv_mean_gaps
    
    # 7. Normalize weights so they average to 1.0 (keeps global learning rate stable)
    normalized_weights = raw_weights / raw_weights.mean()
    
    # 8. Cap the maximum weight to prevent a single extreme outlier from destroying the gradient
    clipped_weights = np.clip(normalized_weights, a_min=0.1, a_max=max_weight_cap)
    
    # 9. Map the final weights back to the training dataframe
    final_weights_dict = clipped_weights.to_dict()
    
    train_df_out = train_df.copy()
    train_df_out['Weight'] = train_df_out['Vehicle_ID'].map(final_weights_dict)
    
    return train_df_out

# Inversely proportionnal to spatial density.
def inverse_space_calculate_pv_weights(train_df, max_weight_cap=5.0):
    """
    Calculates weights directly proportional to the mean spatial gap (in km) 
    around each Probe Vehicle averaged over the entire time horizon.
    """
    # 1. Sort by time and position to order vehicles front-to-back at every single second
    df = train_df.sort_values(by=['Time_s', 'Position_km']).copy()
    
    # 2. Calculate the spatial gap (in km) to the vehicle behind and ahead
    # .diff() looks at the previous row (vehicle behind)
    # .diff(-1) looks at the next row (vehicle ahead)
    df['gap_behind'] = df.groupby('Time_s')['Position_km'].diff()
    df['gap_front'] = df.groupby('Time_s')['Position_km'].diff(-1).abs()
    
    # 3. Handle edge cases (first and last PVs at each second)
    df['gap_behind'] = df['gap_behind'].fillna(df['gap_front'])
    df['gap_front'] = df['gap_front'].fillna(df['gap_behind'])
    
    # 4. Mean spatial gap around the vehicle at this exact second
    df['mean_gap_at_t'] = (df['gap_front'] + df['gap_behind']) / 2.0
    
    # 5. Average this gap over the entire 10 minutes for each specific Vehicle_ID
    pv_mean_gaps = df.groupby('Vehicle_ID')['mean_gap_at_t'].mean()
    
    # 6. Assign raw weights (Directly proportional: larger gap = higher weight)
    raw_weights = 1.0 / (pv_mean_gaps + 1e-6)
    
    # 7. Normalize weights so they average to 1.0 (keeps global learning rate stable)
    normalized_weights = raw_weights / raw_weights.mean()
    
    # 8. Cap the maximum weight to prevent a single extreme outlier from destroying the gradient
    clipped_weights = np.clip(normalized_weights, a_min=0.1, a_max=max_weight_cap)
    
    # 9. Map the final weights back to the training dataframe
    final_weights_dict = clipped_weights.to_dict()
    
    train_df_out = train_df.copy()
    train_df_out['Weight'] = train_df_out['Vehicle_ID'].map(final_weights_dict)
    
    return train_df_out


def calculate_pv_weights(train_df, max_weight_cap=5.0):
    """
    Calculates weights inversely proportional to the mean time gap between PVs
    at fixed 10-meter (0.01 km) spatial milestones.
    """
    # 1. Define the 10m (0.01 km) milestones based on the dataset's physical bounds
    x_min = train_df['Position_km'].min()
    x_max = train_df['Position_km'].max()
    milestones = np.arange(x_min, x_max + 0.01, 0.01) 
    
    crossing_times = []
    
    # 2. Calculate exact crossing times for each PV using linear interpolation
    grouped = train_df.groupby('Vehicle_ID')
    for pv_id, group in grouped:
        # Sort chronologically to ensure trajectory moves forward
        group = group.sort_values('Time_s')
        
        # Drop strictly stationary duplicate positions so numpy interpolation doesn't fail
        group = group.drop_duplicates(subset=['Position_km'], keep='first')
        
        x_pv = group['Position_km'].values
        t_pv = group['Time_s'].values
        
        if len(x_pv) < 2:
            continue # Need at least 2 points to interpolate
            
        # Filter milestones to only those this specific vehicle actually crossed
        valid_milestones = milestones[(milestones >= x_pv.min()) & (milestones <= x_pv.max())]
        
        # Interpolate exact time (Time_s) for each spatial milestone (Position_km)
        # This executes the "uniform repartition of distance traveled during a second"
        t_cross = np.interp(valid_milestones, x_pv, t_pv)
        
        for m, t in zip(valid_milestones, t_cross):
            crossing_times.append({'Milestone': m, 'Vehicle_ID': pv_id, 'Time_s': t})
            
    # 3. Create a dataframe of all milestone crossing events
    cross_df = pd.DataFrame(crossing_times)
    
    # Sort by Milestone, then by Time to order vehicles sequentially at each physical location
    cross_df = cross_df.sort_values(['Milestone', 'Time_s'])
    
    # 4. Calculate Time Gaps at each milestone
    # The vehicle that crossed before is 'in front', the one after is 'behind'
    cross_df['gap_front'] = cross_df.groupby('Milestone')['Time_s'].diff() # t_current - t_front
    cross_df['gap_behind'] = cross_df.groupby('Milestone')['Time_s'].diff(-1).abs() # t_behind - t_current
    
    # Handle edge cases (first vehicle to cross a milestone only has a behind gap, etc.)
    cross_df['gap_front'] = cross_df['gap_front'].fillna(cross_df['gap_behind'])
    cross_df['gap_behind'] = cross_df['gap_behind'].fillna(cross_df['gap_front'])
    
    # Mean time gap at this specific milestone for this specific vehicle
    cross_df['mean_gap_at_m'] = (cross_df['gap_front'] + cross_df['gap_behind']) / 2.0
    
    # 5. Average the gaps across all milestones for each PV
    pv_mean_time_gaps = cross_df.groupby('Vehicle_ID')['mean_gap_at_m'].mean()
    
    # 6. Apply Inverse Weighting: smaller time gap (dense traffic) = higher weight
    # Added tiny epsilon (1e-6) to prevent division by zero if two PVs cross exactly simultaneously
    #raw_weights = 1.0 / (pv_mean_time_gaps + 1e-6)
    raw_weights = pv_mean_time_gaps**3
    print(f"raw_weights : {raw_weights}")
    
    # 7. Normalize weights so they average to 1.0
    normalized_weights = raw_weights / raw_weights.mean()
    
    # 8. Cap maximum weights to preserve gradient stability
    clipped_weights = np.clip(normalized_weights, a_min=0.1, a_max=max_weight_cap)
    
    # 9. Map back to the original training dataframe
    final_weights_dict = clipped_weights.to_dict()
    
    train_df_out = train_df.copy()
    train_df_out['Weight'] = train_df_out['Vehicle_ID'].map(final_weights_dict)
    
    # Fallback to weight 1.0 for any vehicle that didn't cross a milestone (e.g., stationary entire time)
    train_df_out['Weight'] = train_df_out['Weight'].fillna(1.0)
    
    return train_df_out

# ==========================================
# 5. Training without adaptative weights
# ==========================================
def train_pinn(csv_path, t_colloc_all, x_colloc_all, epochs=5000, num_colloc=10000, mu=0.5, gamma=0.05, v_max=50.0, seed = 42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # 1. Prepare Data
    train_df, test_df, t_max, x_max, pv_ids = load_and_split_data(csv_path, pv_ratio=0.10, v_max=v_max, seed = seed)
    
    # Convert training data to tensors
    t_data = torch.tensor(train_df['Time_h'].values, dtype=torch.float32).view(-1, 1).to(device)
    x_data = torch.tensor(train_df['Position_km'].values, dtype=torch.float32).view(-1, 1).to(device)
    rho_data = torch.tensor(train_df['Density'].values, dtype=torch.float32).view(-1, 1).to(device)
    
    # Convert test data for Current Estimation Error (CEE)
    t_test = torch.tensor(test_df['Time_h'].values, dtype=torch.float32).view(-1, 1).to(device)
    x_test = torch.tensor(test_df['Position_km'].values, dtype=torch.float32).view(-1, 1).to(device)
    rho_test = torch.tensor(test_df['Density'].values, dtype=torch.float32).view(-1, 1).to(device)
    
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

        # Extract this epoch's collocation points
        t_colloc = t_colloc_all[epoch]
        x_colloc = x_colloc_all[epoch]
        
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
def train_pinn_with_weights(csv_path, t_colloc_all, x_colloc_all, epochs=5000, num_colloc=10000, mu=0.1, gamma=0.05, v_max=50.0, use_adaptive=False, seed = 42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # 1. Prepare Data
    train_df, test_df, t_max, x_max, pv_ids = load_and_split_data(csv_path, pv_ratio=0.10, v_max=v_max, seed = seed)

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

        # Extract this epoch's collocation points
        t_colloc = t_colloc_all[epoch]
        x_colloc = x_colloc_all[epoch]
        
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
def plot_training_metrics(epochs_logged, loss_total, loss_data, loss_phys, cee, mu, epochs, num_colloc, gamma):
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
    plt.savefig(f"graphs/pinn_metrics_mu_{mu}_epochs_{epochs}_colloc_{num_colloc}_gamma_{gamma}.png", dpi=150)
    plt.show()

# ==========================================
# Visualization Weighted and Unweighted data Function
# ==========================================
def plot_comparative_training_metrics(
    epochs_logged, loss_total, loss_data, loss_phys, cee,
    epochs_logged_adapt, loss_total_adapt, loss_data_adapt, loss_phys_adapt, cee_adapt,
    mu, epochs, num_colloc, gamma
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
    plt.savefig(f"graphs/pinn_comparative_metrics_mu_{mu}_epochs_{epochs}_colloc_{num_colloc}_gamme_{gamma}.png", dpi=150)
    plt.show()

"""
# ==========================================
# Run the Code
# ==========================================
if __name__ == '__main__':
    # Make sure this points to one of the CSVs you generated
    csv_file = "datasets/traffic_N1000_shock.csv" 

    dynamic_weight = True
    
    mu = 0.5
    epochs = 2000
    num_colloc = 20000
    gamma = 0.05
    v_max = 50.0

    
    if os.path.exists(csv_file):

        if not dynamic_weight:
                trained_model, trained_epochs, loss_total, loss_data, loss_phys, cees = train_pinn(
                    csv_file, 
                    epochs, 
                    num_colloc,   # Number of LHS points
                    mu,           # Weighting parameter
                    gamma         # Diffusion parameter
                )
                plot_training_metrics(epochs, loss_total, loss_data, loss_phys, cees, mu, epochs, num_colloc, gamma)
        
        else:
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
                mu, epochs, num_colloc, gamma
            )

    else:
        print(f"Dataset {csv_file} not found. Ensure the generation script was run first.")
"""
        
# ==========================================
# 2. Monte Carlo Evaluation Loop
# ==========================================
def evaluate_adaptive_weights(csv_file, t_colloc_all, x_colloc_all, num_runs=20, epochs=2000):
    results = []
    
    for i in range(num_runs):
        seed = 42 + i * 10 # Change seed for each run
        print(f"\n--- Starting Run {i+1}/{num_runs} (Seed: {seed}) ---")
        
        # 1. Calculate Disparity Index
        _, _, _, _, pv_ids = load_and_split_data(csv_file, seed=seed)
        sorted_pvs = np.sort(pv_ids)
        gaps = np.diff(sorted_pvs)
        disparity_index = np.var(gaps) if len(gaps) > 0 else 0
        
        # 2. Train Base Model
        print("Training Base PINN (Uniform Weights)...")
        _, _, _, _, _, cees_base = train_pinn_with_weights(
            csv_file, t_colloc_all, x_colloc_all, epochs=epochs, use_adaptive=False, seed=seed
        )
        final_cee_base = cees_base[-1]
        
        # 3. Train Adaptive Model
        print("Training Adaptive PINN...")
        _, _, _, _, _, cees_adapt = train_pinn_with_weights(
            csv_file, t_colloc_all, x_colloc_all, epochs=epochs, use_adaptive=True, seed=seed
        )
        final_cee_adapt = cees_adapt[-1]
        
        # 4. Record Results
        cee_improvement = final_cee_base - final_cee_adapt # Positive means adaptive is better
        results.append({
            'Run': i+1,
            'Seed': seed,
            'Disparity_Index': disparity_index,
            'CEE_Base': final_cee_base,
            'CEE_Adapt': final_cee_adapt,
            'CEE_Improvement': cee_improvement
        })
        
    df_results = pd.DataFrame(results)
    df_results.to_csv("adaptive_weights_evaluation.csv", index=False)
    return df_results

def evaluate_adaptive_weights_with_history(csv_file, t_colloc_all, x_colloc_all, num_runs=20, epochs=3000):
    all_cees_base = []
    all_cees_adapt = []
    
    for i in range(num_runs):
        seed = 42 + i * 10
        print(f"\n--- Starting Run {i+1}/{num_runs} (Seed: {seed}) ---")
        
        # Train Base
        _, epochs_logged, _, _, _, cees_base = train_pinn_with_weights(
            csv_file, t_colloc_all, x_colloc_all, epochs=epochs, use_adaptive=False, seed=seed
        )
        
        # Train Adaptive
        _, _, _, _, _, cees_adapt = train_pinn_with_weights(
            csv_file, t_colloc_all, x_colloc_all, epochs=epochs, use_adaptive=True, seed=seed
        )
        
        all_cees_base.append(cees_base)
        all_cees_adapt.append(cees_adapt)
        
    # Convert lists to 2D numpy arrays (Shape: [num_runs, num_logged_epochs])
    return np.array(all_cees_base), np.array(all_cees_adapt), epochs_logged

# ==========================================
# 3. Plotting the Findings
# ==========================================
def plot_evaluation_results(df_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Mean Comparison (Boxplot)
    melted_df = df_results.melt(value_vars=['CEE_Base', 'CEE_Adapt'], 
                                var_name='Model', value_name='Final CEE')
    sns.boxplot(x='Model', y='Final CEE', data=melted_df, ax=ax1, palette="Set2")
    
    # Calculate Statistical Significance
    t_stat, p_val = stats.ttest_rel(df_results['CEE_Base'], df_results['CEE_Adapt'])
    ax1.set_title(f'Overall CEE Comparison (p-value: {p_val:.4f})')
    ax1.set_ylabel('Current Estimation Error (Lower is better)')
    
    # Plot 2: Evolution of Difference vs Disparity
    sns.regplot(x='Disparity_Index', y='CEE_Improvement', data=df_results, ax=ax2, color='darkred')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    # Correlation
    corr, _ = stats.pearsonr(df_results['Disparity_Index'], df_results['CEE_Improvement'])
    ax2.set_title(f'Impact of Disparity on Adaptive Weight Efficacy (r={corr:.2f})')
    ax2.set_xlabel('Disparity Index (Variance of gaps between PVs)')
    ax2.set_ylabel('CEE Improvement (Base CEE - Adapt CEE)')
    
    plt.tight_layout()
    plt.savefig("experimentations/bastien/graphs/adaptive_evaluation_results.png", dpi=150)
    plt.show()


def plot_monte_carlo_cee_evolution(all_cees_base, all_cees_adapt, epochs_logged):
    # Calculate Mean and Standard Deviation across all runs at each epoch step
    mean_base = np.mean(all_cees_base, axis=0)
    std_base = np.std(all_cees_base, axis=0)
    
    mean_adapt = np.mean(all_cees_adapt, axis=0)
    std_adapt = np.std(all_cees_adapt, axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ---------------------------------------------------------
    # Panel 1 (Left): CEE Evolution with Error Bars
    # ---------------------------------------------------------
    # We use capsize to add the flat lines at the end of the error bars
    # Assuming logging happens every 100 epochs, markevery=1 places a marker at each logged step
    ax1.errorbar(epochs_logged, mean_base, yerr=std_base, label='Uniform Weights (Mean ± Std)', 
                 color='black', linestyle='-', marker='o', markersize=4, capsize=3, alpha=0.8)
                 
    ax1.errorbar(epochs_logged, mean_adapt, yerr=std_adapt, label='Adaptive Weights (Mean ± Std)', 
                 color='red', linestyle='--', marker='s', markersize=4, capsize=3, alpha=0.8)
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Current Estimation Error (CEE)', fontsize=12)
    ax1.set_title('CEE Evolution across Monte Carlo Runs', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    
    # ---------------------------------------------------------
    # Panel 2 (Right): Final CEE Distribution
    # ---------------------------------------------------------
    final_base = all_cees_base[:, -1]
    final_adapt = all_cees_adapt[:, -1]
    
    ax2.boxplot([final_base, final_adapt], tick_labels=['Uniform Weights', 'Adaptive Weights'], 
                patch_artist=True, boxprops=dict(facecolor="lightgray"))
                
    ax2.set_ylabel('Final CEE', fontsize=12)
    ax2.set_title('Distribution of Final Validation Error', fontsize=14)
    ax2.grid(True, axis='y', ls="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig("experimentations/bastien/graphs/monte_carlo_cee_errorbars.png", dpi=150)
    plt.show()

# ==========================================
# Run the Code
# ==========================================
if __name__ == '__main__':
    csv_file = "experimentations/bastien/datasets/traffic_N1000_rarefaction.csv" 
    
    if os.path.exists(csv_file):
        # Run different random initializations
        
        num_runs = 10
        epochs = 3000
        num_colloc = 2000

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Quick read to get bounds before generation
        df_temp = pd.read_csv(csv_file)
        t_max_global = (df_temp['Time_s'] / 3600.0).max()
        x_max_global = df_temp['Position_km'].max()
        
        # 1. PRE-COMPUTE ALL EPOCH COLLOCATION POINTS ONCE
        t_colloc_all, x_colloc_all = generate_all_collocation_epochs(
            num_points=num_colloc, 
            num_epochs=epochs, 
            t_max=t_max_global, 
            x_max=x_max_global, 
            device=device
        )

        # Plot only the last epoch
        """
        results_df = evaluate_adaptive_weights(
            csv_file,
            t_colloc_all,
            x_colloc_all,
            num_runs,
            epochs
        )
        """

        # Plot one time with a point at each 100 epochs
        all_cees_base, all_cees_adapt, epochs_logged = evaluate_adaptive_weights_with_history(
            csv_file,
            t_colloc_all,
            x_colloc_all,
            num_runs,
            epochs
        )
        plot_monte_carlo_cee_evolution(all_cees_base, all_cees_adapt, epochs_logged)

    else:
        print(f"Dataset {csv_file} not found.")