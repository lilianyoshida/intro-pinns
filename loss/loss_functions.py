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
