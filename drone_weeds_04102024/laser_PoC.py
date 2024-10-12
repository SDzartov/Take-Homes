import math

# Example: Calculate laser power for 10 meters distance
distance = 10  # meters
intensity_required = 10000  # W/m² (required intensity to burn weed)
focal_length = 800 # Focal length of the camera (in pixels)
real_height = 0.3  # 30 cm tall weed
bbox_height_in_pixels = 100 # Height of the bounding box in the image (in pixels)

class Drone:
    def __init__(self):
        self.position = [0, 0, 0]  # x, y, z coordinates
        self.velocity = [0, 0, 0]  # velocity in x, y, z directions
        self.acceleration = [0, 0, 0]  # acceleration in x, y, z directions
        self.gravity = -9.8 # Gravitational acceleration (m/s^2)
    
    def apply_gravity(self):
        self.acceleration[2] = self.gravity  # Gravity only affects the z-axis

    def update_position(self, time_step):
        # Update velocity based on acceleration
        self.velocity = [v + a * time_step for v, a in zip(self.velocity, self.acceleration)]
        # Update position based on velocity
        self.position = [p + v * time_step for p, v in zip(self.position, self.velocity)]

    def apply_force(self, force):
        # Update acceleration based on force (simplified: F = ma, assuming mass = 1)
        self.acceleration = [f for f in force]

    def move_up(self, target_height, time_step):
            while self.position[2] < target_height:
                # Apply a constant upward force (greater than gravity)
                upward_force = [0, 0, 15]  # Example force to counteract gravity and lift the drone
                self.apply_force(upward_force)

                # Update position based on current velocity and time_step
                self.update_position(time_step)

                print(f"Drone Increasing Height to Maximum Threshold: {drone.position}")
            
            # Once the target height is reached, stop the upward force
            self.apply_force([0, 0, 0])  # Stop applying force when the drone reaches 30m
            print("Target height reached!")


def calculate_distance(drone_position, weed_depth):
    """
    Estimate the distance from the drone to the weed using the weed's depth (distance from the camera).

    Parameters:
    - drone_position: (x, y, z) tuple representing the drone's position.
    - weed_depth: The depth (z-axis distance) from the drone's camera to the weed.

    Returns:
    - distance: The Euclidean distance between the drone and the weed in meters.
    """
    
    # The weed's position relative to the drone
    weed_x = 0  # Assuming weed is directly ahead (x-axis is aligned)
    weed_y = 0  # Assuming no vertical offset (y-axis aligned)
    weed_z = weed_depth  # The depth of the weed in front of the drone (z-axis)
    
    # Get the drone's position (x, y, z)
    drone_x, drone_y, drone_z = drone_position
    
    # Calculate the Euclidean distance between the drone and the weed
    distance = math.sqrt(
        (weed_x - drone_x) ** 2 +
        (weed_y - drone_y) ** 2 +
        (weed_z - drone_z) ** 2
    )
    
    return distance

def estimate_depth_from_bbox(focal_length, real_height, bbox_height_in_pixels):
    """
    Estimate the depth of the weed from its bounding box height.

    Parameters:
    - focal_length: Focal length of the camera (in pixels or mm).
    - real_height: The real-world height of the weed (in meters).
    - bbox_height_in_pixels: The height of the bounding box in pixels.

    Returns:
    - depth: Estimated depth (distance from the camera) in meters.
    """
    
    depth = (focal_length * real_height) / bbox_height_in_pixels
    return depth

def calculate_laser_power(distance, intensity_required, initial_radius=0.001, divergence_angle=0.01):
    # Calculate beam radius at the given distance
    beam_radius = initial_radius + distance * math.tan(divergence_angle)
    
    # Calculate the beam area at the given distance
    beam_area = math.pi * (beam_radius ** 2)
    
    # Calculate the required power to maintain the desired intensity
    required_power = intensity_required * beam_area
    
    return required_power

weed_depth = estimate_depth_from_bbox(focal_length,real_height,bbox_height_in_pixels)

# Simulate the drone in a basic environment
drone = Drone()
time_step = 0.1  # seconds
print(f"Launching Drone: {drone.position}")

#drone.apply_gravity()
for step in range(10):
    if step == 0:
        drone.move_up(15, 0.1)
    drone.update_position(time_step)
    distance_to_weed = calculate_distance(drone.position, weed_depth)
    laser_power = calculate_laser_power(distance_to_weed, intensity_required)
    
    print(f"Step {step}: Drone Position: {drone.position}")
    print(f"Distance to Weed: {distance_to_weed} meters")
    print(f"Required Laser Power at {distance_to_weed} meters: {laser_power:.2f} W")