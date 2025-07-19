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
