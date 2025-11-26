import math
import random

def generate_planets(filename="planets.txt"):
    """
    Generate a planet system similar to Rutherford atomic model.
    One massive central body with smaller planets orbiting around it.
    """
    
    with open(filename, 'w') as f:
        # Central massive body (like the nucleus)
        # Position: origin, no velocity, no acceleration
        central_mass = 500.0
        central_radius = 10.0
        f.write(f"0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 {central_mass} {central_radius} 255 215 0 255\n")
        
        # Generate orbiting planets (like electrons)
        # Create orbital shells with evenly distributed planets
        # Each shell is tilted at a different angle for visual interest
        shells = [
            {"radius": 300, "count": 4, "tilt_axis": (1, 0, 0), "tilt_angle": 0, "color": (255, 100, 100)},      # Inner: Red
            {"radius": 400, "count": 6, "tilt_axis": (0, 1, 0), "tilt_angle": math.pi/4, "color": (100, 150, 255)},   # Middle: Blue
            {"radius": 500, "count": 4, "tilt_axis": (1, 1, 0), "tilt_angle": math.pi/3, "color": (100, 255, 150)}    # Outer: Green
        ]
        
        planet_idx = 0
        for shell in shells:
            orbit_radius = shell["radius"]
            count = shell["count"]
            tilt_axis = shell["tilt_axis"]
            tilt_angle = shell["tilt_angle"]
            shell_color = shell["color"]
            
            # Normalize tilt axis
            axis_length = math.sqrt(sum(a*a for a in tilt_axis))
            if axis_length > 0:
                tilt_axis = tuple(a / axis_length for a in tilt_axis)
            
            # Evenly distribute planets around the orbit
            for j in range(count):
                theta = (2 * math.pi * j) / count  # Evenly spaced angles
            
                # Position on orbit (starting in xy-plane)
                x = orbit_radius * math.cos(theta)
                y = orbit_radius * math.sin(theta)
                z = 0.0
                
                # Apply rotation around tilt axis using Rodrigues' rotation formula
                if tilt_angle != 0:
                    k = tilt_axis  # rotation axis
                    pos = [x, y, z]
                    
                    # v_rot = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
                    cos_a = math.cos(tilt_angle)
                    sin_a = math.sin(tilt_angle)
                    
                    # k · v
                    k_dot_v = k[0]*pos[0] + k[1]*pos[1] + k[2]*pos[2]
                    
                    # k × v
                    k_cross_v = [
                        k[1]*pos[2] - k[2]*pos[1],
                        k[2]*pos[0] - k[0]*pos[2],
                        k[0]*pos[1] - k[1]*pos[0]
                    ]
                    
                    # Apply rotation
                    x = pos[0]*cos_a + k_cross_v[0]*sin_a + k[0]*k_dot_v*(1-cos_a)
                    y = pos[1]*cos_a + k_cross_v[1]*sin_a + k[1]*k_dot_v*(1-cos_a)
                    z = pos[2]*cos_a + k_cross_v[2]*sin_a + k[2]*k_dot_v*(1-cos_a)
                
                # Calculate orbital velocity (perpendicular to position vector in the tilted plane)
                # v = sqrt(GM/r) for circular orbit
                G = 5000  # Gravitational constant (adjusted for simulation scale)
                orbital_speed = math.sqrt(G * central_mass / orbit_radius)
                
                # Velocity perpendicular to position in original xy-plane
                vx_orig = -orbital_speed * math.sin(theta)
                vy_orig = orbital_speed * math.cos(theta)
                vz_orig = 0.0
                
                # Apply same rotation to velocity vector
                if tilt_angle != 0:
                    vel = [vx_orig, vy_orig, vz_orig]
                    k_dot_v = k[0]*vel[0] + k[1]*vel[1] + k[2]*vel[2]
                    k_cross_v = [
                        k[1]*vel[2] - k[2]*vel[1],
                        k[2]*vel[0] - k[0]*vel[2],
                        k[0]*vel[1] - k[1]*vel[0]
                    ]
                    vx = vel[0]*cos_a + k_cross_v[0]*sin_a + k[0]*k_dot_v*(1-cos_a)
                    vy = vel[1]*cos_a + k_cross_v[1]*sin_a + k[1]*k_dot_v*(1-cos_a)
                    vz = vel[2]*cos_a + k_cross_v[2]*sin_a + k[2]*k_dot_v*(1-cos_a)
                else:
                    vx = vx_orig
                    vy = vy_orig
                    vz = vz_orig
                
                # No initial acceleration (will be calculated by simulation)
                ax = ay = az = 0.0
                
                # Small planet properties (consistent within each shell)
                mass = 10.0
                radius = 5.0
                
                # Use dedicated color for this shell
                col_r, col_g, col_b = shell_color
                col_a = 255
                
                f.write(f"{x:.2f} {y:.2f} {z:.2f} {vx:.2f} {vy:.2f} {vz:.2f} {ax:.2f} {ay:.2f} {az:.2f} {mass} {radius} {col_r} {col_g} {col_b} {col_a}\n")
                planet_idx += 1
    
    print(f"Generated {sum(s['count'] for s in shells) + 1} planets in '{filename}'")
    print(f"- 1 central massive body (mass={central_mass})")
    print(f"- Inner shell: 4 planets at radius 100")
    print(f"- Middle shell: 6 planets at radius 200")
    print(f"- Outer shell: 4 planets at radius 300")

if __name__ == "__main__":
    generate_planets("planets.txt")