

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
