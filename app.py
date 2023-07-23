import streamlit as st
import os
from PIL import Image
import cv2
import json
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn as nn
from model import LogisticSiameseRegression, SiameseResNet, SpatialAttention
from model import preprocess_image
import pytesseract
# import easyocr
# The following line is for Windows users only. You may need to change the path below depending on the path of your Tesseract OCR installation.
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Move the processed image to a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define the directory to store uploaded images
UPLOAD_FOLDER = 'static/uploads'
TEST_FOLDER = 'static/test'

JSON_FILE = "check_template_resize.json"
THRESHOLD = 0.4 

# reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory
# Load the JSON data
def load_json_data(json_file_path):
    with open(json_file_path) as json_file:
        json_data = json.load(json_file)
    return json_data


siamese_model = SiameseResNet()
siamese_model = nn.DataParallel(siamese_model).to(device)
siamese_model.load_state_dict(torch.load("resnet_with_attention_triplet_saturday.pth", map_location=device))

model_rms = LogisticSiameseRegression(siamese_model).to(device)
model_rms.load_state_dict(torch.load("model_final_attention_saturday.pth", map_location=device))



def get_genuine_and_test_images_for_class(class_images, test_image):
    test_images = [test_image] * len(class_images)
    class_images_df = pd.DataFrame({'GenuineImages': class_images, 'TestImage': test_images})
    return class_images_df



transformation = transforms.Compose([
    transforms.Resize((155,220)),
    transforms.ToTensor(),
])


class GenuineTestDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        genuine_image_path = row['GenuineImages']
        test_image_path = row['TestImage']

        genuine_image = Image.open(genuine_image_path)
        test_image = Image.open(test_image_path)

        genuine_image = preprocess_image(genuine_image)
        test_image = preprocess_image(test_image)
        images = [genuine_image, test_image]
        st.image(images,width=200)
        if self.transform is not None:
            genuine_image = self.transform(genuine_image)
            test_image = self.transform(test_image)

        return genuine_image, test_image



# Extract the signatures from the cheque image
def extract_signature(image_path, json_data):
    signatures = []
    # Iterate over the files in the JSON
    for file_data in json_data.values():
        filename = file_data['filename']
        st.image(image_path)
        cheque_image = cv2.imread(image_path)
        if cheque_image is None:
            raise ValueError("Failed to read the image:", image_path)
        cheque_image = cv2.resize(cheque_image, (1125, 525)) # Resizing the cheque to template size

        # Iterate over the regions in the file
        for region in file_data['regions']:
            region_name = region['region_attributes']['name']
            x = region['shape_attributes']['x']
            y = region['shape_attributes']['y']
            width = region['shape_attributes']['width']
            height = region['shape_attributes']['height']
            
            # Extract the region from the cheque image
            region_image = cheque_image[y:y+height, x:x+width]
            gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            if "Signature" in region_name:
                # Find the bounding box coordinates of the non-white pixels
                # coords = cv2.findNonZero(binary)
                # x, y, w, h = cv2.boundingRect(coords)

                # # Crop the image using the modified bounding box coordinates
                # cropped_image = region_image[y:y+h, x:x+w]
                signature_path = f"static/test/test_signature_{len(signatures)}.jpg"
                cv2.imwrite(signature_path, region_image)
                signatures.append(signature_path)

            if "MCIR" in region_name:
                text = pytesseract.image_to_string(binary, lang='mcr')

            text = pytesseract.image_to_string(binary, lang='eng')
            # text = reader.readtext(binary,detail = 0)
            st.write(f"{region_name}: {text}")
    return signatures


# Prepare the dataset for signature forgery detection
def prepare_dataset(genuine_images, test_image):
    test_images = [test_image] * len(genuine_images)
    df = pd.DataFrame({'GenuineImages': genuine_images, 'TestImage': test_images})
    return df


# Perform signature forgery detection
def perform_voting(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    positive_votes = 0
    negative_votes = 0
    print("*****************PREDICTIONS**********************")
    # Iterate over the dataset
    for i, data in enumerate(dataloader, 0):
        genuine_image, test_image = data
        genuine_image = genuine_image.to(device)
        test_image = test_image.to(device)

        with torch.no_grad():
            output = model(genuine_image, test_image)
            percent = (1-output) * 100
            st.write(f"Similarity with image {i+1}: {int(percent)}%")
            if output < THRESHOLD:
                positive_votes += 1
            else:
                negative_votes += 1
    print(f"Positive Votes: {positive_votes}\t Negative votes: {negative_votes}")
    return positive_votes, negative_votes


# Display uploaded images
def display_uploaded_images(uploaded_images):
    if len(uploaded_images) > 0:
        st.subheader("Uploaded Images:")
        num_images = len(uploaded_images)
        num_columns = 3  # Define the number of columns in each row

        # Calculate the number of rows required
        num_rows = (num_images + num_columns - 1) // num_columns

        # Display the images in multiple rows
        for i in range(num_rows):
            row_images = uploaded_images[i * num_columns: (i + 1) * num_columns]
            st.image(row_images, width=220)  # Set the desired width of the images


# Main function
def main():
    st.title('Signature Extraction and Forgery Detection')

    # Load JSON data
    json_data = load_json_data(JSON_FILE)

    uploaded_images = []
    test_images = []

    for i in range(1, 4):
        image_file = st.file_uploader(f'Upload Genuine Image {i}')
        if image_file is not None:
            file_path = os.path.join(UPLOAD_FOLDER, image_file.name).replace("\\", "/")
            image = Image.open(image_file)
            image.save(file_path)
            uploaded_images.append(file_path)

    test_file = st.file_uploader('Upload Check Image')
    if test_file is not None:
        file_path = os.path.join(TEST_FOLDER, test_file.name).replace("\\", "/")
        image = Image.open(test_file)
        image.save(file_path)
        signatures = extract_signature(file_path,json_data)
        test_images.extend(signatures)

    if len(uploaded_images) < 3 or len(test_images) < 1:
        st.error("Upload 3 genuine images and 1 test image")
        return  # Exit the function if images are not uploaded properly

    if len(test_images) == 0:
        st.error("Test image not uploaded")
        return  # Exit the function if test image is not uploaded


    # Display uploaded images
    display_uploaded_images(uploaded_images)
    print("--------------------------------------START----------------------------------\n")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='color: #55ff33; font-size: 28px; text-align:center;'>PREDICTION</p>", unsafe_allow_html=True)
    for test_image in test_images:
        st.markdown("<hr>", unsafe_allow_html=True)
        df = get_genuine_and_test_images_for_class(uploaded_images, test_image)
        try:
            dataset = GenuineTestDataset(df, transform=transformation)
            dataloader = DataLoader(dataset, batch_size=1)

            pos, neg = perform_voting(model_rms, dataloader)
            if neg == 0:
                final_prediction = "Genuine"  # for genuine
            else:
                final_prediction = "Forged"  # for forged
        except Exception as e:
            st.error("Error performing voting: " + str(e))
        # uploaded_images.clear()
        if final_prediction == "Genuine":
            st.markdown("<p style='color: #46da28; font-size: 20px;'>Final prediction: Genuine</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #DE3163; font-size: 20px;'>Final prediction: Forged</p>", unsafe_allow_html=True)
        print(f"Final Prediction: {final_prediction}")
        st.image(test_image, width=300)  # Set the desired width of the image
    print("\n-------------------------------------END-----------------------------------\n")

if __name__ == '__main__':
    main()
