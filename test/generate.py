import math
import random

def generate_orbital_system(filename, num_orbits=5, planets_per_orbit=8):
    """
    Generate a stable planetary system with:
    - 1-2 massive central bodies
    - Multiple orbital rings with planets
    - Symmetric placement for stability
    """
    
    with open(filename, 'w') as f:
        # Central massive star (stationary)
        central_mass = 1000000.0
        f.write(f"0.0 0.0 0.0 ")  # position
        f.write(f"0.0 0.0 0.0 ")  # velocity
        f.write(f"0.0 0.0 0.0 ")  # acceleration
        f.write(f"{central_mass} 50.0 ")  # mass, radius
        f.write(f"255 255 100 255\n")  # yellow star
        
        # Optional: Add a second smaller central star for binary system
        if random.random() > 0.5:
            offset = 150.0
            binary_mass = 500000.0
            # Circular orbit around common center
            orbital_v = math.sqrt(6.674e-11 * (central_mass + binary_mass) / (2 * offset))
            orbital_v *= 1e-6  # scale factor
            
            f.write(f"{offset} 0.0 0.0 ")  # position
            f.write(f"0.0 {orbital_v} 0.0 ")  # velocity
            f.write(f"0.0 0.0 0.0 ")  # acceleration
            f.write(f"{binary_mass} 35.0 ")  # mass, radius
            f.write(f"255 200 150 255\n")  # orange star
        
        # Generate orbital rings
        G = 6.674e-11
        colors = [
            (100, 150, 255, 255),  # blue
            (100, 255, 150, 255),  # green
            (255, 100, 150, 255),  # pink
            (255, 150, 100, 255),  # orange
            (200, 100, 255, 255),  # purple
            (255, 255, 150, 255),  # light yellow
            (150, 255, 255, 255),  # cyan
            (255, 150, 255, 255),  # magenta
        ]
        
        for orbit_idx in range(num_orbits):
            # Orbital radius increases with each ring
            radius = 300.0 + orbit_idx * 200.0
            
            # Calculate orbital velocity for circular orbit
            # v = sqrt(G * M / r)
            orbital_speed = math.sqrt(G * central_mass / radius)
            # Scale down for simulation units
            orbital_speed *= 1e-6
            
            # Add slight variation to prevent perfect resonance
            orbital_speed *= (0.95 + random.random() * 0.1)
            
            # Planet properties for this orbit
            planet_mass = 1000.0 * (1.0 - orbit_idx * 0.1)
            planet_radius = 8.0 - orbit_idx * 0.5
            color = colors[orbit_idx % len(colors)]
            
            # Place planets symmetrically around the orbit
            for planet_idx in range(planets_per_orbit):
                angle = 2 * math.pi * planet_idx / planets_per_orbit
                
                # Add small random offset for visual interest (±5%)
                r = radius * (0.95 + random.random() * 0.1)
                
                # Position
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                z = random.uniform(-20, 20)  # slight 3D variation
                
                # Velocity (perpendicular to radius for circular orbit)
                vx = -orbital_speed * math.sin(angle)
                vy = orbital_speed * math.cos(angle)
                vz = 0.0
                
                # Small random velocity perturbation for realism
                vx += random.uniform(-0.001, 0.001)
                vy += random.uniform(-0.001, 0.001)
                
                # Write planet data
                f.write(f"{x:.2f} {y:.2f} {z:.2f} ")
                f.write(f"{vx:.6f} {vy:.6f} {vz:.6f} ")
                f.write(f"0.0 0.0 0.0 ")
                f.write(f"{planet_mass:.1f} {planet_radius:.1f} ")
                f.write(f"{color[0]} {color[1]} {color[2]} {color[3]}\n")
        
        # Add a few random asteroids in eccentric orbits
        num_asteroids = 10
        for i in range(num_asteroids):
            # Random elliptical orbit parameters
            semi_major = random.uniform(200, 1500)
            eccentricity = random.uniform(0.3, 0.7)
            angle = random.uniform(0, 2 * math.pi)
            
            # Position at perihelion or aphelion
            r = semi_major * (1 - eccentricity)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            z = random.uniform(-50, 50)
            
            # Velocity for elliptical orbit
            v = math.sqrt(G * central_mass * (2/r - 1/semi_major))
            v *= 1e-6
            vx = -v * math.sin(angle)
            vy = v * math.cos(angle)
            vz = random.uniform(-0.001, 0.001)
            
            # Small asteroid properties
            asteroid_mass = random.uniform(10, 100)
            asteroid_radius = random.uniform(2, 4)
            
            f.write(f"{x:.2f} {y:.2f} {z:.2f} ")
            f.write(f"{vx:.6f} {vy:.6f} {vz:.6f} ")
            f.write(f"0.0 0.0 0.0 ")
            f.write(f"{asteroid_mass:.1f} {asteroid_radius:.1f} ")
            f.write(f"200 200 200 255\n")  # gray asteroids

# Generate different demo files
print("Generating demo files...")

# 1. Simple system with 5 rings, 8 planets each
generate_orbital_system("solar_system_demo.txt", num_orbits=5, planets_per_orbit=8)
print(f"Created solar_system_demo.txt")

# 2. Dense system with more planets
generate_orbital_system("dense_system_demo.txt", num_orbits=6, planets_per_orbit=10)
print(f"Created dense_system_demo.txt")

# 3. Sparse system with fewer planets
generate_orbital_system("sparse_system_demo.txt", num_orbits=8, planets_per_orbit=5)
print(f"Created sparse_system_demo.txt")

print("\nAll demo files generated successfully!")
print("Each file contains 50-70 planets in stable orbits around a central star.")
print("The systems should remain stable and not become chaotic.")