#!/usr/bin/env python3
"""
Physics-Informed Neural Network for ALINE Electromagnetic Wave Propagation
Implements the first study: E_field_asy from COMSOL model

Based on ALINE_3D_90deg_compactH.m electromagnetic wave physics:
- Frequency: 25 MHz
- Magnetized plasma with anisotropic dielectric tensor
- 3D cylindrical geometry with coaxial excitation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as sp
from scipy.spatial import distance
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ALINEPhysics:
    """
    ALINE plasma physics parameters and equations
    Based on COMSOL model parameters
    """
    def __init__(self):
        # Model parameters from COMSOL
        self.f0 = 25e6  # [Hz] fundamental frequency
        self.B0 = 0.091  # [T] magnetic field
        self.mi = 40 * 1.67e-27  # [kg] helium ion mass
        self.me = 9.1e-31  # [kg] electron mass
        self.w0 = 2 * np.pi * self.f0  # [rad/s] angular frequency
        self.theta_deg = 1.0  # [deg] B field tilt angle
        self.theta = self.theta_deg * np.pi / 180  # [rad]
        self.v_m = 1e8  # [rad/s] electron-neutron collision frequency
        self.nuiN = 1e6  # [Hz] ion-neutron collision frequency
        self.n_den = 2.48e17  # [1/m^3] density factor
        self.sigma_factor = 1e-3  # sigma_perp / sigma_par
        
        # Physical constants
        self.c0 = 3e8  # [m/s] speed of light
        self.eps0 = 8.854e-12  # [F/m] vacuum permittivity
        self.mu0 = 4*np.pi*1e-7  # [H/m] vacuum permeability
        self.e = 1.602e-19  # [C] elementary charge
        self.k_B = 1.381e-23  # [J/K] Boltzmann constant
        
        # Derived parameters
        self.k0 = self.w0 / self.c0  # vacuum wavenumber
        self.lambda0 = 2 * np.pi / self.k0  # vacuum wavelength
        
        # Geometry parameters (from COMSOL model)
        self.R_chamber = 0.15  # [m] chamber radius
        self.L_chamber = 0.5   # [m] chamber length
        self.R_electrode = 0.05  # [m] electrode radius
        self.h_electrode = 0.02  # [m] electrode height
        
    def plasma_frequencies(self, n_e, n_i):
        """Calculate plasma frequencies"""
        wpe = np.sqrt(n_e * self.e**2 / (self.eps0 * self.me))  # electron plasma freq
        wpi = np.sqrt(n_i * self.e**2 / (self.eps0 * self.mi))  # ion plasma freq
        return wpe, wpi
    
    def cyclotron_frequencies(self):
        """Calculate cyclotron frequencies"""
        wce = -self.e * self.B0 / self.me  # electron cyclotron freq (negative)
        wci = self.e * self.B0 / self.mi   # ion cyclotron freq
        return wce, wci
    
    def stix_parameters(self, n_e, n_i):
        """Calculate Stix dielectric tensor parameters"""
        wpe, wpi = self.plasma_frequencies(n_e, n_i)
        wce, wci = self.cyclotron_frequencies()
        
        # Stix parameters with collisions
        w_complex = self.w0 + 1j * self.v_m
        
        P_stix = 1 - (wpe**2 + wpi**2) / w_complex**2
        R_stix = 1 - wpe**2/(w_complex*(w_complex + wce)) - wpi**2/(w_complex*(w_complex + wci))
        L_stix = 1 - wpe**2/(w_complex*(w_complex - wce)) - wpi**2/(w_complex*(w_complex - wci))
        
        S_stix = (R_stix + L_stix) / 2
        D_stix = (R_stix - L_stix) / 2
        
        return P_stix, R_stix, L_stix, S_stix, D_stix
    
    def dielectric_tensor(self, x, y, z):
        """
        Calculate 3x3 dielectric tensor at given position
        Following COMSOL model variable definitions
        """
        # Density profile (simplified Gaussian for electrode region)
        r_electrode = np.sqrt(y**2 + z**2)
        if isinstance(r_electrode, (int, float)):
            r_electrode = np.array([r_electrode])
        
        # Density profile: higher density near electrode
        n_profile = self.n_den * np.exp(-r_electrode**2 / (2 * self.R_electrode**2))
        n_e = n_i = n_profile
        
        # Calculate Stix parameters
        P_stix, R_stix, L_stix, S_stix, D_stix = self.stix_parameters(n_e, n_i)
        
        # Trigonometric functions for B-field orientation
        sin_theta = np.sin(self.theta)
        cos_theta = np.cos(self.theta)
        sin2_theta = sin_theta**2
        cos2_theta = cos_theta**2
        cos_sin_theta = cos_theta * sin_theta
        
        # Dielectric tensor elements (from COMSOL variables)
        eps11 = S_stix * sin2_theta + P_stix * cos2_theta
        eps12 = 1j * D_stix * sin_theta
        eps13 = -S_stix * cos_sin_theta + P_stix * cos_sin_theta
        eps21 = -1j * D_stix * sin_theta
        eps22 = S_stix
        eps23 = 1j * D_stix * cos_theta
        eps31 = -S_stix * cos_sin_theta + P_stix * cos_sin_theta
        eps32 = -1j * D_stix * cos_theta
        eps33 = S_stix * cos2_theta + P_stix * sin2_theta
        
        # Return as 3x3 tensor
        epsilon = np.array([[eps11, eps12, eps13],
                           [eps21, eps22, eps23],
                           [eps31, eps32, eps33]], dtype=complex)
        
        return epsilon
    
    def get_plasma_dielectric_for_loss(self, X):
        """
        Calculate plasma dielectric tensor for loss computation (simplified)
        Returns effective dielectric constant for wave equation
        """
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        
        # Simplified: use only the plasma frequency effect
        r_electrode = torch.sqrt(y**2 + z**2)
        n_profile = self.n_den * torch.exp(-r_electrode**2 / (2 * self.R_electrode**2))
        
        # Plasma frequency
        wpe = torch.sqrt(n_profile * self.e**2 / (self.eps0 * self.me))
        
        # Simplified dielectric: eps = 1 - wpe^2/w^2
        eps_plasma = 1 - wpe**2 / self.w0**2
        
        # Ensure minimum value to avoid instabilities
        eps_effective = torch.clamp(eps_plasma, min=0.1)
        
        return eps_effective

class ALINEGeometry:
    """
    ALINE 3D geometry definition
    Based on COMSOL model geometry
    """
    def __init__(self, physics):
        self.physics = physics
        self.create_geometry()
    
    def create_geometry(self):
        """Define computational domain boundaries"""
        # Chamber boundaries (cylindrical)
        self.x_min, self.x_max = -0.25, 0.25  # [m]
        self.y_min, self.y_max = -0.15, 0.15  # [m] 
        self.z_min, self.z_max = -0.15, 0.15  # [m]
        
        # Store chamber dimensions for easy access
        self.L_chamber = self.x_max - self.x_min  # Total length
        self.R_chamber = self.physics.R_chamber   # Radius
        
        # Electrode position (at y=0, rotated 90 degrees)
        self.electrode_center = np.array([0.0, 0.0, 0.0])
        
    def is_inside_chamber(self, x, y, z):
        """Check if point is inside chamber (cylindrical)"""
        r_chamber = np.sqrt(y**2 + z**2)
        return (r_chamber <= self.physics.R_chamber) & \
               (x >= self.x_min) & (x <= self.x_max)
    
    def is_on_electrode(self, x, y, z, tol=1e-3):
        """Check if point is on electrode surface (90-degree rotated disk)"""
        # Electrode is rotated 90 degrees, so it's in the x-z plane at y=0
        r_electrode = np.sqrt((x - self.electrode_center[0])**2 + 
                             (z - self.electrode_center[2])**2)
        on_disk = (r_electrode <= self.physics.R_electrode) & \
                  (np.abs(y - self.electrode_center[1]) <= self.physics.h_electrode/2)
        return on_disk
    
    def is_on_chamber_wall(self, x, y, z, tol=1e-3):
        """Check if point is on chamber wall"""
        r_chamber = np.sqrt(y**2 + z**2)
        on_cylinder = np.abs(r_chamber - self.physics.R_chamber) < tol
        on_endcaps = (np.abs(x - self.x_min) < tol) | (np.abs(x - self.x_max) < tol)
        return on_cylinder | on_endcaps
    
    def is_on_coax_port(self, x, y, z, tol=1e-3):
        """Check if point is on coaxial port (simplified as small circle)"""
        # Simplified: port at x_max, center
        on_port = (np.abs(x - self.x_max) < tol) & \
                  (np.sqrt(y**2 + z**2) <= 0.01)  # Small port radius
        return on_port

class EMWaveNet(nn.Module):
    """
    Neural network for electromagnetic wave fields
    Outputs: [Ex_real, Ex_imag, Ey_real, Ey_imag, Ez_real, Ez_imag]
    """
    def __init__(self, layers=[3, 64, 64, 64, 64, 6], activation_type='tanh'):
        super(EMWaveNet, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        # Choose activation function
        if activation_type == 'tanh':
            self.activation = nn.Tanh()
        elif activation_type == 'sine':
            self.activation = lambda x: torch.sin(30 * x)  # Sine activation for periodic solutions
        elif activation_type == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)  # Swish activation
        else:
            self.activation = nn.Tanh()
            
        self.init_weights()
    
    def init_weights(self):
        """Xavier initialization"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x):
        """Forward pass: x = [x, y, z]"""
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)  # No activation on output
        return x

class ALINESampler:
    """
    Sampling points for PINN training
    """
    def __init__(self, geometry, n_domain=5000, n_boundary=1000, n_port=200):
        self.geometry = geometry
        self.n_domain = n_domain
        self.n_boundary = n_boundary
        self.n_port = n_port
        
        self.generate_sampling_points()
    
    def generate_sampling_points(self):
        """Generate sampling points for domain and boundaries"""
        # Domain points (inside chamber, not on electrode)
        self.domain_points = self._sample_domain()
        
        # Boundary points (chamber walls, excluding electrode and port)
        self.boundary_points = self._sample_boundaries()
        
        # Port points (coaxial excitation)
        self.port_points = self._sample_port()
        
        # Convert to tensors
        self.X_domain = torch.tensor(self.domain_points, dtype=torch.float32, requires_grad=True, device=device)
        self.X_boundary = torch.tensor(self.boundary_points, dtype=torch.float32, requires_grad=True, device=device)
        self.X_port = torch.tensor(self.port_points, dtype=torch.float32, requires_grad=True, device=device)
    
    def _sample_domain(self):
        """Sample points inside computational domain"""
        points = []
        count = 0
        max_attempts = self.n_domain * 10
        
        while count < self.n_domain and max_attempts > 0:
            # Random sampling in bounding box
            x = np.random.uniform(self.geometry.x_min, self.geometry.x_max)
            y = np.random.uniform(self.geometry.y_min, self.geometry.y_max)  
            z = np.random.uniform(self.geometry.z_min, self.geometry.z_max)
            
            # Check if inside chamber and not on electrode
            if (self.geometry.is_inside_chamber(x, y, z) and 
                not self.geometry.is_on_electrode(x, y, z) and
                not self.geometry.is_on_chamber_wall(x, y, z)):
                points.append([x, y, z])
                count += 1
            
            max_attempts -= 1
        
        return np.array(points)
    
    def _sample_boundaries(self):
        """Sample points on chamber boundaries (PEC surfaces)"""
        points = []
        
        # Cylindrical wall
        n_cyl = int(0.7 * self.n_boundary)
        for _ in range(n_cyl):
            theta = np.random.uniform(0, 2*np.pi)
            x = np.random.uniform(self.geometry.x_min, self.geometry.x_max)
            y = self.geometry.physics.R_chamber * np.cos(theta)
            z = self.geometry.physics.R_chamber * np.sin(theta)
            points.append([x, y, z])
        
        # End caps
        n_caps = self.n_boundary - n_cyl
        for _ in range(n_caps):
            r = np.random.uniform(0, self.geometry.physics.R_chamber)
            theta = np.random.uniform(0, 2*np.pi)
            x = np.random.choice([self.geometry.x_min, self.geometry.x_max])
            y = r * np.cos(theta)
            z = r * np.sin(theta)
            if not self.geometry.is_on_electrode(x, y, z):
                points.append([x, y, z])
        
        return np.array(points)
    
    def _sample_port(self):
        """Sample points on coaxial port"""
        points = []
        
        for _ in range(self.n_port):
            r = np.random.uniform(0, 0.01)  # Small port radius
            theta = np.random.uniform(0, 2*np.pi)
            x = self.geometry.x_max
            y = r * np.cos(theta)
            z = r * np.sin(theta)
            points.append([x, y, z])
        
        return np.array(points)

class ALINELoss:
    """
    Loss functions for ALINE PINN
    """
    def __init__(self, physics, geometry):
        self.physics = physics
        self.geometry = geometry
    
    def maxwell_residual(self, X, E_pred):
        """
        Calculate Maxwell equation residual for electromagnetic waves
        Wave equation: ∇²E + k₀²εᵣE = 0
        """
        # Extract real and imaginary parts
        Ex_real, Ex_imag = E_pred[:, 0:1], E_pred[:, 1:2]
        Ey_real, Ey_imag = E_pred[:, 2:3], E_pred[:, 3:4]
        Ez_real, Ez_imag = E_pred[:, 4:5], E_pred[:, 5:6]
        
        # Calculate Laplacian of E components
        try:
            lap_Ex_real = self._laplacian_from_coords(Ex_real, X)
            lap_Ex_imag = self._laplacian_from_coords(Ex_imag, X)
            lap_Ey_real = self._laplacian_from_coords(Ey_real, X)
            lap_Ey_imag = self._laplacian_from_coords(Ey_imag, X)
            lap_Ez_real = self._laplacian_from_coords(Ez_real, X)
            lap_Ez_imag = self._laplacian_from_coords(Ez_imag, X)
        except Exception as e:
            print(f"Laplacian computation failed: {e}")
            # Return a small loss to continue training
            return torch.tensor(1e-6, device=X.device, requires_grad=True)
        
        # Plasma dielectric effect (simplified)
        k02 = self.physics.k0**2
        try:
            eps_r = self.physics.get_plasma_dielectric_for_loss(X)
        except:
            eps_r = 1.0  # Fallback to vacuum
        
        # Wave equation residual: ∇²E + k₀²εᵣE = 0
        residual_Ex_real = lap_Ex_real + k02 * eps_r * Ex_real
        residual_Ex_imag = lap_Ex_imag + k02 * eps_r * Ex_imag
        residual_Ey_real = lap_Ey_real + k02 * eps_r * Ey_real
        residual_Ey_imag = lap_Ey_imag + k02 * eps_r * Ey_imag
        residual_Ez_real = lap_Ez_real + k02 * eps_r * Ez_real
        residual_Ez_imag = lap_Ez_imag + k02 * eps_r * Ez_imag
        
        # Calculate mean squared residual
        loss_x = torch.mean(residual_Ex_real**2 + residual_Ex_imag**2)
        loss_y = torch.mean(residual_Ey_real**2 + residual_Ey_imag**2)
        loss_z = torch.mean(residual_Ez_real**2 + residual_Ez_imag**2)
        
        return loss_x + loss_y + loss_z
    
    def _laplacian_from_coords(self, f, X):
        """Calculate Laplacian ∇²f using coordinate tensor X"""
        # First derivatives
        fx = self._gradient(f, X)[:, 0:1]  # df/dx
        fy = self._gradient(f, X)[:, 1:2]  # df/dy  
        fz = self._gradient(f, X)[:, 2:3]  # df/dz
        
        # Second derivatives
        fxx = self._gradient(fx, X)[:, 0:1]  # d²f/dx²
        fyy = self._gradient(fy, X)[:, 1:2]  # d²f/dy²
        fzz = self._gradient(fz, X)[:, 2:3]  # d²f/dz²
        
        return fxx + fyy + fzz
    
    def _gradient(self, f, X):
        """Calculate gradient df/dX where X is the full coordinate tensor"""
        try:
            grad = torch.autograd.grad(f, X, 
                                     grad_outputs=torch.ones_like(f),
                                     create_graph=True, retain_graph=True,
                                     allow_unused=True)[0]
            if grad is None:
                # If gradient is None, return zeros with same shape as X
                return torch.zeros_like(X)
            return grad
        except RuntimeError as e:
            print(f"Gradient computation failed: {e}")
            return torch.zeros_like(X)
    
    def boundary_loss_pec(self, X_boundary, E_pred):
        """
        Perfect Electric Conductor boundary condition: n × E = 0
        For simplicity, enforce E_tangential = 0
        """
        # For PEC surfaces, tangential E-field components should be zero
        # This is a simplified implementation
        E_magnitude = torch.sqrt(torch.sum(E_pred**2, dim=1))
        return torch.mean(E_magnitude**2)
    
    def port_loss(self, X_port, E_pred, V_port=115.9):
        """
        Coaxial port excitation: integrate E·dl = V
        Simplified implementation
        """
        # For coaxial port, impose voltage constraint
        # This is a placeholder for proper port implementation
        E_avg = torch.mean(E_pred, dim=0)
        target = torch.tensor([V_port, 0, 0, 0, 0, 0], device=device, dtype=torch.float32)
        return torch.mean((E_avg - target)**2)

class ALINETrainer:
    """
    PINN trainer for ALINE electromagnetic wave problem
    """
    def __init__(self, model, physics, geometry, sampler, 
                 lr=1e-3, weight_pde=1.0, weight_bc=10.0, weight_port=5.0):
        self.model = model.to(device)
        self.physics = physics
        self.geometry = geometry
        self.sampler = sampler
        self.loss_calculator = ALINELoss(physics, geometry)
        
        # Initial loss weights (will be adapted during training)
        self.weight_pde_initial = weight_pde
        self.weight_bc_initial = weight_bc
        self.weight_port_initial = weight_port
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        
        # Training history
        self.history = {'total_loss': [], 'pde_loss': [], 'bc_loss': [], 'port_loss': []}
    
    def adaptive_loss_weights(self, epoch):
        """
        Dynamically adjust loss weights during training
        Focus on boundaries first, then gradually emphasize physics
        """
        if epoch < 1000:
            # Stage 1: Focus on boundary conditions
            w_pde = 0.1 * self.weight_pde_initial
            w_bc = 10.0 * self.weight_bc_initial  
            w_port = 5.0 * self.weight_port_initial
        elif epoch < 3000:
            # Stage 2: Balanced training
            w_pde = 1.0 * self.weight_pde_initial
            w_bc = 5.0 * self.weight_bc_initial
            w_port = 2.0 * self.weight_port_initial
        else:
            # Stage 3: Emphasize physics
            w_pde = 5.0 * self.weight_pde_initial
            w_bc = 1.0 * self.weight_bc_initial
            w_port = 1.0 * self.weight_port_initial
            
        return w_pde, w_bc, w_port
    
    def train(self, epochs=10000, print_every=1000):
        """Train the PINN model"""
        self.model.train()
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Get adaptive loss weights
            weight_pde, weight_bc, weight_port = self.adaptive_loss_weights(epoch)
            
            # Forward pass
            E_domain = self.model(self.sampler.X_domain)
            E_boundary = self.model(self.sampler.X_boundary)
            E_port = self.model(self.sampler.X_port)
            
            # Calculate losses
            pde_loss = self.loss_calculator.maxwell_residual(self.sampler.X_domain, E_domain)
            bc_loss = self.loss_calculator.boundary_loss_pec(self.sampler.X_boundary, E_boundary)
            port_loss = self.loss_calculator.port_loss(self.sampler.X_port, E_port)
            
            # Total loss with adaptive weights
            total_loss = (weight_pde * pde_loss + 
                         weight_bc * bc_loss + 
                         weight_port * port_loss)
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Store history
            self.history['total_loss'].append(total_loss.item())
            self.history['pde_loss'].append(pde_loss.item())
            self.history['bc_loss'].append(bc_loss.item())
            self.history['port_loss'].append(port_loss.item())
            
            # Print progress with current weights
            if epoch % print_every == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:6d} | Total: {total_loss.item():.3e} | "
                      f"PDE: {pde_loss.item():.3e} (w={weight_pde:.1f}) | "
                      f"BC: {bc_loss.item():.3e} (w={weight_bc:.1f}) | "
                      f"Port: {port_loss.item():.3e} (w={weight_port:.1f}) | "
                      f"Time: {elapsed:.1f}s")
            
            # Learning rate decay
            if epoch % 1000 == 0 and epoch > 0:
                self.scheduler.step()
    
    def plot_training_history(self):
        """Plot training loss curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(len(self.history['total_loss']))
        
        ax1.semilogy(epochs, self.history['total_loss'])
        ax1.set_title('Total Loss')
        ax1.set_xlabel('Epoch')
        ax1.grid(True)
        
        ax2.semilogy(epochs, self.history['pde_loss'])
        ax2.set_title('PDE Loss')
        ax2.set_xlabel('Epoch')
        ax2.grid(True)
        
        ax3.semilogy(epochs, self.history['bc_loss'])
        ax3.set_title('Boundary Loss')
        ax3.set_xlabel('Epoch')
        ax3.grid(True)
        
        ax4.semilogy(epochs, self.history['port_loss'])
        ax4.set_title('Port Loss')
        ax4.set_xlabel('Epoch')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()

def visualize_results(model, geometry, physics, nx=50, ny=50):
    """
    Visualize the electromagnetic field solution
    """
    model.eval()
    
    # Create evaluation grid (2D slice at z=0)
    x_eval = np.linspace(geometry.x_min, geometry.x_max, nx)
    y_eval = np.linspace(geometry.y_min, geometry.y_max, ny)
    X_grid, Y_grid = np.meshgrid(x_eval, y_eval)
    Z_grid = np.zeros_like(X_grid)
    
    # Flatten for evaluation
    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()
    Z_flat = Z_grid.flatten()
    
    # Filter points inside domain
    valid_mask = []
    for i in range(len(X_flat)):
        valid_mask.append(geometry.is_inside_chamber(X_flat[i], Y_flat[i], Z_flat[i]) and
                         not geometry.is_on_electrode(X_flat[i], Y_flat[i], Z_flat[i]))
    
    valid_mask = np.array(valid_mask)
    X_valid = X_flat[valid_mask]
    Y_valid = Y_flat[valid_mask]
    Z_valid = Z_flat[valid_mask]
    
    # Evaluate model
    with torch.no_grad():
        X_tensor = torch.tensor(np.column_stack([X_valid, Y_valid, Z_valid]), 
                               dtype=torch.float32).to(device)
        E_pred = model(X_tensor).cpu().numpy()
    
    # Calculate field magnitude
    E_magnitude = np.sqrt(np.sum(E_pred**2, axis=1))
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Electric field magnitude
    scatter1 = ax1.scatter(X_valid, Y_valid, c=E_magnitude, cmap='viridis', s=1)
    ax1.set_title('Electric Field Magnitude |E|')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1, label='|E| [V/m]')
    
    # Add geometry outline
    theta = np.linspace(0, 2*np.pi, 100)
    chamber_x = geometry.physics.R_chamber * np.cos(theta)
    chamber_y = geometry.physics.R_chamber * np.sin(theta)
    ax1.plot(chamber_x, chamber_y, 'r-', linewidth=2, label='Chamber wall')
    
    # Electrode outline (simplified)
    electrode_x = np.array([-geometry.physics.R_electrode, geometry.physics.R_electrode, 
                           geometry.physics.R_electrode, -geometry.physics.R_electrode, 
                           -geometry.physics.R_electrode])
    electrode_y = np.array([-geometry.physics.h_electrode/2, -geometry.physics.h_electrode/2,
                           geometry.physics.h_electrode/2, geometry.physics.h_electrode/2,
                           -geometry.physics.h_electrode/2])
    ax1.plot(electrode_x, electrode_y, 'k-', linewidth=3, label='Electrode')
    ax1.legend()
    
    # Real part of Ex
    Ex_real = E_pred[:, 0]
    scatter2 = ax2.scatter(X_valid, Y_valid, c=Ex_real, cmap='RdBu_r', s=1)
    ax2.set_title('Real(Ex)')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2, label='Re(Ex) [V/m]')
    ax2.plot(chamber_x, chamber_y, 'k-', linewidth=1)
    ax2.plot(electrode_x, electrode_y, 'k-', linewidth=2)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main execution function
    """
    print("ALINE PINN Electromagnetic Wave Solver")
    print("=====================================")
    
    # Initialize physics and geometry
    physics = ALINEPhysics()
    geometry = ALINEGeometry(physics)
    
    print(f"Frequency: {physics.f0/1e6:.1f} MHz")
    print(f"Wavelength: {physics.lambda0:.3f} m")
    print(f"Chamber dimensions: R={geometry.R_chamber:.3f} m, L={geometry.L_chamber:.3f} m")
    
    # Generate sampling points
    print("\nGenerating sampling points...")
    sampler = ALINESampler(geometry, n_domain=2000, n_boundary=500, n_port=100)
    print(f"Domain points: {len(sampler.domain_points)}")
    print(f"Boundary points: {len(sampler.boundary_points)}")
    print(f"Port points: {len(sampler.port_points)}")
    
    # Initialize neural network with improved architecture
    print("\nInitializing neural network...")
    model = EMWaveNet(layers=[3, 128, 128, 128, 128, 6], activation_type='tanh')  # Deeper network
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize trainer with adaptive weights
    print("\nInitializing trainer...")
    trainer = ALINETrainer(model, physics, geometry, sampler, 
                          lr=1e-3, weight_pde=1.0, weight_bc=1.0, weight_port=1.0)  # Equal initial weights
    
    # Train the model
    print("\nStarting training...")
    trainer.train(epochs=5000, print_every=500)  # Reduced epochs for testing
    
    # Plot training history
    print("\nPlotting training history...")
    trainer.plot_training_history()
    
    # Visualize results
    print("\nVisualizing results...")
    visualize_results(model, geometry, physics, nx=30, ny=30)  # Reduced resolution
    
    print("\nTraining completed!")

# Execute the main function
if __name__ == "__main__":
    main()