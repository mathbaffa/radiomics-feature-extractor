import os
import csv
import concurrent.futures
import six
import SimpleITK as sitk
from radiomics import featureextractor

# Set PyRadiomics paramameters
params = 'Params.yaml' 
extractor = featureextractor.RadiomicsFeatureExtractor(params)

# Load images and save features as a csv file
sample = "SynD_96_022" # Name of the sample/patient
input_dir = sample + '/'  # Directory containing the images
output_csv = sample + '.csv'  # Path to output the CSV file

# Set directory of the mask. Can be used a single mask for whole-slide images or an directory containing multiple 
mask = 'mask.png'

num_files_in_folder = len(os.listdir(input_dir))
iterator_count = 0

def process_image(image_path):
    image = sitk.ReadImage(image_path) # Load the image using SimpleITK
    
    gray_image = sitk.VectorIndexSelectionCast(image, 0) # Convert the image to grayscale
    
    result = extractor.execute(gray_image, mask, 255)  # Use gray_image as the image for feature extraction
    
    features = list(six.itervalues(result)) # Get the feature values

    return features


with open(output_csv, 'w', newline='') as csvfile:
    
    writer = csv.writer(csvfile)
    
    num_cores = os.cpu_count() # Get the number of processor cores
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            
            # Process images in parallel using all available processor cores
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:

                #Print progress in percentage
                percentage = (iterator_count * 100) / num_files_in_folder
                print(f"Processing {percentage:.2f}")
                iterator_count += 1
                
                # Extract the features
                feature_future = executor.submit(process_image, image_path)
                features = feature_future.result()

                # Save the features into a csv file
                writer.writerow(features)
                
