import streamlit as st
import os
from PIL import Image
from torchvision.transforms import transforms
import torch
from torch.utils.data import DataLoader
from model import SiameseResNet, LogisticSiameseRegression
from model import preprocess_image
import pandas as pd
import torch.nn as nn


# Define the directory to store uploaded images
UPLOAD_FOLDER = 'static/uploads'
TEST_FOLDER = 'static/test'

# Move the processed image to a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the JSON data


siamese_model = SiameseResNet()
siamese_model = nn.DataParallel(siamese_model).to(device)
siamese_model.load_state_dict(torch.load("resnet_with_attention_triplet_saturday.pth", map_location=device))

model_rms = LogisticSiameseRegression(siamese_model).to(device)
model_rms.load_state_dict(torch.load("model_final_attention_saturday.pth", map_location=device))

transformation = transforms.Compose([
    transforms.Resize((310,440)),
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


def get_genuine_and_test_images_for_class(class_images, test_image):
    test_images = [test_image] * len(class_images)
    class_images_df = pd.DataFrame({'GenuineImages': class_images, 'TestImage': test_images})
    return class_images_df


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
            st.write(f"Similarity with Image {i+1}: {int(percent)}%")
            if output < 0.5:
                positive_votes += 1
            else:
                negative_votes += 1
    print(f"Positive Votes: {positive_votes}\t Negative votes: {negative_votes}")
    return positive_votes, negative_votes



def main():
    st.title('Signature Forgery Detection')
    uploaded_images = []
    test_images = []

    for i in range(1, 4):
        image_file = st.file_uploader(f'Upload Genuine Image {i}')
        if image_file is not None:
            file_path = os.path.join(UPLOAD_FOLDER, image_file.name).replace("\\", "/")
            image = Image.open(image_file)
            image.save(file_path)
            uploaded_images.append(file_path)

    test_file = st.file_uploader('Upload Test Image')
    if test_file is not None:
        file_path = os.path.join(TEST_FOLDER, test_file.name).replace("\\", "/")
        image = Image.open(test_file)
        image.save(file_path)
        test_images.append(file_path)  # Use append instead of extend

    if len(uploaded_images) < 3 or len(test_images) < 1:
        st.error("Upload 3 genuine images and 1 test image")
        return  # Exit the function if images are not uploaded properly

    if len(test_images) == 0:
        st.error("Test image not uploaded")
        return  # Exit the function if test image is not uploaded

    # Display uploaded images
    if len(uploaded_images) > 0:
        st.subheader("Uploaded Images:")
        num_images = len(uploaded_images)
        num_columns = 3  # Define the number of columns in each row

        # Calculate the number of rows required
        num_rows = (num_images + num_columns - 1) // num_columns

        # Display the images in multiple rows
        for i in range(num_rows):
            row_images = uploaded_images[i * num_columns: (i + 1) * num_columns]
            st.image(row_images, width=200)  # Set the desired width of the images

    df = get_genuine_and_test_images_for_class(uploaded_images, test_images[0])
    print(df)
    final_prediction = ""  # Assign a default value
    try:
        dataset = GenuineTestDataset(df, transform=transformation)
        dataloader = DataLoader(dataset, batch_size=1)

        pos, neg = perform_voting(model_rms, dataloader)
        if neg == 0:
        # if pos > neg:
            final_prediction = "Genuine"  # for genuine
        else:
            final_prediction = "Forged"  # for forged
    except Exception as e:
        st.error("Error performing voting: " + str(e))

    uploaded_images.clear()
    if final_prediction == "Genuine":
        st.markdown("<p style='color: green; font-size: 24px;'>Final prediction: Genuine</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: red; font-size: 24px;'>Final prediction: Forged</p>", unsafe_allow_html=True)
    # st.dataframe(df)
    print(f"Final Prediction {final_prediction}")

    if len(test_images) > 0:
        st.subheader("Test Image:")
        for image_path in test_images:
            print(f"Image Path: {image_path}")
            st.image(image_path, width=200)  # Set the desired width of the image


if __name__ == '__main__':
    main()
