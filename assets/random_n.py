import math
import random

def generate_random_planets(filename, num_planets):
    """
    Generate a random planetary system with completely random positions and velocities.
    Creates chaotic but interesting simulations.
    """
    
    with open(filename, 'w') as f:
        for i in range(num_planets):
            # Random position in a large cube
            x = random.uniform(-2000, 2000)
            y = random.uniform(-2000, 2000)
            z = random.uniform(-2000, 2000)
            
            # Random velocity
            vx = random.uniform(-0.1, 0.1)
            vy = random.uniform(-0.1, 0.1)
            vz = random.uniform(-0.1, 0.1)
            
            # Acceleration starts at 0
            ax = 0.0
            ay = 0.0
            az = 0.0
            
            # Random mass and radius
            # Most planets are small, but some are larger
            if random.random() < 0.95:
                # Regular planet
                mass = random.uniform(100, 5000)
                radius = random.uniform(3, 10)
            else:
                # Occasional heavy planet
                mass = random.uniform(10000, 100000)
                radius = random.uniform(15, 40)
            
            # Random color
            col_r = random.randint(50, 255)
            col_g = random.randint(50, 255)
            col_b = random.randint(50, 255)
            col_a = 255
            
            # Write to file
            f.write(f"{x:.2f} {y:.2f} {z:.2f} ")
            f.write(f"{vx:.6f} {vy:.6f} {vz:.6f} ")
            f.write(f"{ax:.6f} {ay:.6f} {az:.6f} ")
            f.write(f"{mass:.1f} {radius:.1f} ")
            f.write(f"{col_r} {col_g} {col_b} {col_a}\n")

def generate_clustered_planets(filename, num_planets):
    """
    Generate planets in random clusters with some having orbital motion.
    More interesting than pure random.
    """
    
    with open(filename, 'w') as f:
        # Determine number of clusters (roughly 1 cluster per 20 planets)
        num_clusters = max(1, num_planets // 20)
        
        # Generate cluster centers
        clusters = []
        for _ in range(num_clusters):
            cx = random.uniform(-1500, 1500)
            cy = random.uniform(-1500, 1500)
            cz = random.uniform(-1500, 1500)
            cluster_mass = random.uniform(50000, 500000)
            clusters.append((cx, cy, cz, cluster_mass))
        
        G = 6.674e-11
        
        # Assign planets to clusters
        planets_per_cluster = num_planets // num_clusters
        remaining = num_planets % num_clusters
        
        for cluster_idx, (cx, cy, cz, cluster_mass) in enumerate(clusters):
            # How many planets in this cluster
            n_planets = planets_per_cluster + (1 if cluster_idx < remaining else 0)
            
            for _ in range(n_planets):
                # Random offset from cluster center
                distance = random.uniform(50, 500)
                theta = random.uniform(0, 2 * math.pi)
                phi = random.uniform(0, math.pi)
                
                x = cx + distance * math.sin(phi) * math.cos(theta)
                y = cy + distance * math.sin(phi) * math.sin(theta)
                z = cz + distance * math.cos(phi)
                
                # Give some orbital velocity around cluster center
                if random.random() < 0.7:  # 70% have orbital motion
                    # Calculate orbital velocity
                    v = math.sqrt(G * cluster_mass / distance) * 1e-6
                    # Perpendicular direction to radius
                    vx = -v * math.sin(theta)
                    vy = v * math.cos(theta)
                    vz = random.uniform(-0.01, 0.01)
                else:
                    # Random velocity
                    vx = random.uniform(-0.05, 0.05)
                    vy = random.uniform(-0.05, 0.05)
                    vz = random.uniform(-0.05, 0.05)
                
                # Random mass and radius
                if random.random() < 0.98:
                    mass = random.uniform(100, 3000)
                    radius = random.uniform(3, 8)
                else:
                    mass = random.uniform(5000, 50000)
                    radius = random.uniform(12, 30)
                
                # Random color
                col_r = random.randint(50, 255)
                col_g = random.randint(50, 255)
                col_b = random.randint(50, 255)
                col_a = 255
                
                f.write(f"{x:.2f} {y:.2f} {z:.2f} ")
                f.write(f"{vx:.6f} {vy:.6f} {vz:.6f} ")
                f.write(f"0.0 0.0 0.0 ")
                f.write(f"{mass:.1f} {radius:.1f} ")
                f.write(f"{col_r} {col_g} {col_b} {col_a}\n")

# Generate files with different planet counts
print("Generating random planet files...")

sizes = [50, 100, 500, 1000]

for size in sizes:
    # Pure random
    filename = f"random_{size}.txt"
    generate_random_planets(filename, size)
    print(f"Created {filename} - {size} planets with random positions/velocities")
    
    # Clustered (more interesting)
    filename = f"clustered_{size}.txt"
    generate_clustered_planets(filename, size)
    print(f"Created {filename} - {size} planets in clusters")

print("\nAll files generated successfully!")
print("\nFiles created:")
print("  random_*.txt - Pure random positions and velocities")
print("  clustered_*.txt - Planets grouped in clusters with some orbital motion")