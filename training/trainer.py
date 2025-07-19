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
