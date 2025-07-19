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
