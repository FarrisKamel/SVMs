import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import scipy.io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Configuration
image_size = (128, 128)  # Resize images to 128x128

# Function to preprocess images and extract HOG features
def preprocess_images(folder_path, label):
    features = []
    labels = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            # Load image
            image = imread(file_path)
            
            #Check if image is greyscale
            if len(image.shape) != 3 or image.shape[2] != 3:
                gray_image = image
                resized_image = resize(gray_image, image_size, anti_aliasing=True)
            else:
                # Convert to grayscale and resize
                gray_image = rgb2gray(image)
                resized_image = resize(gray_image, image_size, anti_aliasing=True)

            #Flatten the image
            flattened_features = resized_image.flatten()

            features.append(flattened_features)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    return np.array(features), np.array(labels)

########### Raw Images ######################
# Paths to all datasets
class_paths = {
    "fire_truck": "data/072.fire-truck",
#    "beer_mug": "data/010.beer-mug",
#    "bonsai_101": "data/015.bonsai-101",
#    "bulldozer": "data/023.bulldozer",
#    "cactus": "data/025.cactus",
#     "cake": "data/026.cake",
#     "canoe": "data/030.canoe",
     "coffee_mug": "data/041.coffee-mug",
#     "dog": "data/056.dog",
#     "dolphin_101": "data/057.dolphin-101",
#     "duck": "data/060.duck",
#     "electric_guitar_101": "data/063.electric-guitar-101",
#     "fern": "data/068.fern",
#     "fighter_jet": "data/069.fighter-jet",
#     "goldfish": "data/087.goldfish",
#     "goose": "data/089.goose",
#     "hamburger": "data/095.hamburger",
#     "harp": "data/098.harp",
#     "helicopter_101": "data/102.helicopter-101",
#     "horse": "data/105.horse",
#     "hot_dog": "data/108.hot-dog",
#     "ice_cream_cone": "data/115.ice-cream-cone",
#     "kayak": "data/122.kayak",
#     "killer_whale": "data/124.killer-whale",
#     "leopards_101": "data/129.leopards-101",
#     "light_house": "data/132.light-house",
#     "mandolin": "data/136.mandolin",
#     "motorbikes_101": "data/145.motorbikes-101",
#     "school_bus": "data/178.school-bus",
#     "smokestack": "data/188.smokestack",
#     "snowmobile": "data/192.snowmobile",
#     "spaghetti": "data/196.spaghetti",
#     "speed_boat": "data/197.speed-boat",
#     "sushi": "data/206.sushi",
#     "swan": "data/207.swan",
#     "windmill": "data/245.windmill",
#     "zebra": "data/250.zebra",
#     "airplanes_101": "data/251.airplanes-101",
#     "car_side_101": "data/252.car-side-101",
#     "faces_easy_101": "data/253.faces-easy-101",
#    "clutter": "data/257.clutter",
}

# Preprocess firetruck dataset (label=1)
firetruck_features, firetruck_labels = preprocess_images(class_paths["fire_truck"], label=1)

# Preprocess all other datasets (label=0)
non_firetruck_features_list = []
non_firetruck_labels_list = []
for class_name, folder_path in class_paths.items():
    if class_name != "fire_truck":  # Skip the firetruck class
        print(f"Processing {folder_path} as non-firetruck (label=0)...")
        features, labels = preprocess_images(folder_path, label=0)
        if features.size > 0:  # Only add non-empty arrays
            non_firetruck_features_list.append(features)
            non_firetruck_labels_list.append(labels)
        else:
            print(f"Skipping empty dataset: {folder_path}")


# Combine firetruck and non-firetruck datasets
if len(non_firetruck_features_list) > 0:
    non_firetruck_features = np.vstack(non_firetruck_features_list)
    non_firetruck_labels = np.hstack(non_firetruck_labels_list)
else:
    raise ValueError("No valid non-firetruck datasets found!")

combined_features = np.vstack((firetruck_features, non_firetruck_features))
combined_labels = np.hstack((firetruck_labels, non_firetruck_labels))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_features, combined_labels, test_size=0.2, random_state=42)

# Train SVM classifier
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# Test the model
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Output results
print("Raw Images Result")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

# Flot all the features to compare
pca = PCA(n_components=2)
features_2d = pca.fit_transform(np.vstack((firetruck_features, non_firetruck_features)))
labels = np.hstack(([1] * firetruck_features.shape[0], [0] * non_firetruck_features.shape[0]))

# Plot the features
plt.scatter(features_2d[labels == 1, 0], features_2d[labels == 1, 1], label="Firetruck", alpha=0.5)
plt.scatter(features_2d[labels == 0, 0], features_2d[labels == 0, 1], label="Coffee Mug", alpha=0.5)
plt.legend()
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Raw Image Feature Distribution")
plt.show()



########### LBP Images ######################
# Paths to all LBPsets
fire_truck_data = scipy.io.loadmat("LBP/072.fire-truck.mat")
non_fire_truck_data = scipy.io.loadmat("LBP/041.coffee-mug.mat")

# Extract features and labels
firetruck_features = fire_truck_data['feature']
firetruck_labels = [1] * firetruck_features.shape[0]  # Firetruck is label 1

non_fire_truck_features = non_fire_truck_data['feature']
speed_boat_labels = [0] * non_fire_truck_features.shape[0]  # Speed boat is label 0

# Combine firetruck and speed boat datasets
combined_features = np.vstack((firetruck_features, non_fire_truck_features))
combined_labels = np.hstack((firetruck_labels, speed_boat_labels))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    combined_features, combined_labels, test_size=0.2, random_state=42
)

# Train SVM classifier
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# Test the model
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Output results
print("LBP")
print("SVM Results:")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

# Flot all the features to compare
pca = PCA(n_components=2)
features_2d = pca.fit_transform(np.vstack((firetruck_features, non_fire_truck_features)))
labels = np.hstack(([1] * firetruck_features.shape[0], [0] * non_fire_truck_features.shape[0]))

# Plot the features
plt.scatter(features_2d[labels == 1, 0], features_2d[labels == 1, 1], label="Firetruck", alpha=0.5)
plt.scatter(features_2d[labels == 0, 0], features_2d[labels == 0, 1], label="Coffee Mug", alpha=0.5)
plt.legend()
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("LBP Feature Distribution")
plt.show()



########### PHOG Images ######################
# Paths to all LBPsets
fire_truck_data = scipy.io.loadmat("PHOG/072.fire-truck.mat")
non_fire_truck_data = scipy.io.loadmat("PHOG/041.coffee-mug.mat")

# Extract features and labels
firetruck_features = fire_truck_data['feature']
firetruck_labels = [1] * firetruck_features.shape[0]  # Firetruck is label 1

non_fire_truck_features = non_fire_truck_data['feature']
speed_boat_labels = [0] * non_fire_truck_features.shape[0]  # Speed boat is label 0
print(firetruck_features.shape)
print(non_fire_truck_features.shape)

# Combine firetruck and speed boat datasets
combined_features = np.vstack((firetruck_features, non_fire_truck_features))
combined_labels = np.hstack((firetruck_labels, speed_boat_labels))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    combined_features, combined_labels, test_size=0.2, random_state=42
)

# Train SVM classifier
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# Test the model
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Output results
print("PHOG")
print("SVM Results:")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

# Flot all the features to compare
pca = PCA(n_components=2)
features_2d = pca.fit_transform(np.vstack((firetruck_features, non_fire_truck_features)))
labels = np.hstack(([1] * firetruck_features.shape[0], [0] * non_fire_truck_features.shape[0]))

# Plot the features
plt.scatter(features_2d[labels == 1, 0], features_2d[labels == 1, 1], label="Firetruck", alpha=0.5)
plt.scatter(features_2d[labels == 0, 0], features_2d[labels == 0, 1], label="Coffee Mug", alpha=0.5)
plt.legend()
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("LBP Feature Distribution")
plt.show()



########### SIFT Images ######################
# Paths to all SIFTsets
fire_truck_data = scipy.io.loadmat("SIFT/072.fire-truck.mat")
non_fire_truck_data = scipy.io.loadmat("SIFT/041.coffee-mug.mat")

# Extract features and labels
firetruck_features = fire_truck_data['feature']
firetruck_labels = [1] * firetruck_features.shape[0]  # Firetruck is label 1

non_fire_truck_features = non_fire_truck_data['feature']
speed_boat_labels = [0] * non_fire_truck_features.shape[0]  # Non-firetruck is label 0

# Combine firetruck and speed boat datasets
combined_features = np.vstack((firetruck_features, non_fire_truck_features))
combined_labels = np.hstack((firetruck_labels, speed_boat_labels))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    combined_features, combined_labels, test_size=0.2, random_state=42
)

# Train SVM classifier
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# Test the model
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Output results
print("SIFT")
print("SVM Results:")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

# Flot all the features to compare
pca = PCA(n_components=2)
features_2d = pca.fit_transform(np.vstack((firetruck_features, non_fire_truck_features)))
labels = np.hstack(([1] * firetruck_features.shape[0], [0] * non_fire_truck_features.shape[0]))

# Plot the features
plt.scatter(features_2d[labels == 1, 0], features_2d[labels == 1, 1], label="Firetruck", alpha=0.5)
plt.scatter(features_2d[labels == 0, 0], features_2d[labels == 0, 1], label="Coffee Mug", alpha=0.5)
plt.legend()
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("LBP Feature Distribution")
plt.show()



########### RECOV Images ######################
# Paths to all RECOV sets
fire_truck_data = scipy.io.loadmat("RECOV/072.fire-truck.mat")
non_fire_truck_data = scipy.io.loadmat("RECOV/041.coffee-mug.mat")

# Extract features and labels
firetruck_features = fire_truck_data['feature']
firetruck_labels = [1] * firetruck_features.shape[0]  # Firetruck is label 1

non_fire_truck_features = non_fire_truck_data['feature']
speed_boat_labels = [0] * non_fire_truck_features.shape[0]  # Non-firetruck is label 0

# Combine firetruck and speed boat datasets
combined_features = np.vstack((firetruck_features, non_fire_truck_features))
combined_labels = np.hstack((firetruck_labels, speed_boat_labels))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    combined_features, combined_labels, test_size=0.2, random_state=42
)

# Train SVM classifier
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# Test the model
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Output results
print("RECOV")
print("SVM Results:")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

# Flot all the features to compare
pca = PCA(n_components=2)
features_2d = pca.fit_transform(np.vstack((firetruck_features, non_fire_truck_features)))
labels = np.hstack(([1] * firetruck_features.shape[0], [0] * non_fire_truck_features.shape[0]))

# Plot the features
plt.scatter(features_2d[labels == 1, 0], features_2d[labels == 1, 1], label="Firetruck", alpha=0.5)
plt.scatter(features_2d[labels == 0, 0], features_2d[labels == 0, 1], label="Coffee Mug", alpha=0.5)
plt.legend()
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("LBP Feature Distribution")
plt.show()


########### All Features ###############

# LBP
LBP_fire_truck_data = scipy.io.loadmat("LBP/072.fire-truck.mat")
LBP_non_fire_truck_data = scipy.io.loadmat("LBP/041.coffee-mug.mat")

LBP_firetruck_features = fire_truck_data['feature']
LBP_firetruck_labels = [1] * firetruck_features.shape[0]  # Firetruck is label 1

LBP_non_fire_truck_features = non_fire_truck_data['feature']
LBP_speed_boat_labels = [0] * non_fire_truck_features.shape[0]  # Speed boat is label 0

LBP_combined_features = np.vstack((LBP_firetruck_features, LBP_non_fire_truck_features))
combined_labels = np.hstack((LBP_firetruck_labels, LBP_speed_boat_labels))

# PHOG
PHOG_fire_truck_data = scipy.io.loadmat("PHOG/072.fire-truck.mat")
PHOG_non_fire_truck_data = scipy.io.loadmat("PHOG/041.coffee-mug.mat")

PHOG_firetruck_features = fire_truck_data['feature']
PHOG_firetruck_labels = [1] * firetruck_features.shape[0]  # Firetruck is label 1

PHOG_non_fire_truck_features = non_fire_truck_data['feature']
PHOG_speed_boat_labels = [0] * non_fire_truck_features.shape[0]  # Speed boat is label 0

PHOG_combined_features = np.vstack((PHOG_firetruck_features, PHOG_non_fire_truck_features))
combined_labels = np.hstack((PHOG_firetruck_labels, PHOG_speed_boat_labels))

# SIFT 
SIFT_fire_truck_data = scipy.io.loadmat("SIFT/072.fire-truck.mat")
SIFT_non_fire_truck_data = scipy.io.loadmat("SIFT/041.coffee-mug.mat")

SIFT_firetruck_features = fire_truck_data['feature']
SIFT_firetruck_labels = [1] * firetruck_features.shape[0]  # Firetruck is label 1

SIFT_non_fire_truck_features = non_fire_truck_data['feature']
SIFT_speed_boat_labels = [0] * non_fire_truck_features.shape[0]  # Non-firetruck is label 0

SIFT_combined_features = np.vstack((SIFT_firetruck_features, SIFT_non_fire_truck_features))
combined_labels = np.hstack((SIFT_firetruck_labels, SIFT_speed_boat_labels))

# RECOV
RECOV_fire_truck_data = scipy.io.loadmat("RECOV/072.fire-truck.mat")
RECOV_non_fire_truck_data = scipy.io.loadmat("RECOV/041.coffee-mug.mat")

RECOV_firetruck_features = fire_truck_data['feature']
RECOV_firetruck_labels = [1] * firetruck_features.shape[0]  # Firetruck is label 1

RECOV_non_fire_truck_features = non_fire_truck_data['feature']
RECOV_speed_boat_labels = [0] * non_fire_truck_features.shape[0]  # Non-firetruck is label 0

RECOV_combined_features = np.vstack((RECOV_firetruck_features, RECOV_non_fire_truck_features))
combined_labels = np.hstack((RECOV_firetruck_labels, RECOV_speed_boat_labels))

# Combine all features
all_firetruck_features = np.concatenate((LBP_combined_features, PHOG_combined_features, SIFT_combined_features, RECOV_combined_features), axis=1)
print(all_firetruck_features.shape)
print(combined_labels.shape)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    all_firetruck_features, combined_labels, test_size=0.2, random_state=42
)

# Train SVM classifier
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# Test the model
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Output results
print("All Combined")
print("SVM Results:")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)



