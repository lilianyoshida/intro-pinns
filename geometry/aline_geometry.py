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
