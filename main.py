from physics.aline_physics import ALINEPhysics
from geometry.aline_geometry import ALINEGeometry
from data.sampler import ALINESampler
from models.em_wave_net import EMWaveNet
from training.trainer import ALINETrainer
from visualize import visualize_results

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