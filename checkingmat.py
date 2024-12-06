import scipy.io

# Replace these with the paths to your .mat files
fire_truck_mat_path = "LBP/072.fire-truck.mat"
speed_boat_mat_path = "LBP/197.speed-boat.mat"

# Load the .mat files
fire_truck_data = scipy.io.loadmat(fire_truck_mat_path)
speed_boat_data = scipy.io.loadmat(speed_boat_mat_path)

# Inspect the keys in both files
print("Keys in fire truck .mat file:", fire_truck_data.keys())
print("Keys in speed boat .mat file:", speed_boat_data.keys())

# Example: Inspect the feature key if it exists
if 'feature' in fire_truck_data:
    print("Fire truck feature shape:", fire_truck_data['feature'].shape)
if 'feature' in speed_boat_data:
    print("Speed boat feature shape:", speed_boat_data['feature'].shape)

