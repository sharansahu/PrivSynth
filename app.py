import streamlit as st
import pandas as pd
import sqlite3
import hashlib

from dpwgan.datasets import CategoricalDataset
from dpwgan.utils import create_categorical_gan
from dpwgan.synthetic_data_statistics import *

# Security
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# DB Management
conn = sqlite3.connect('data.db', check_same_thread=False)
c = conn.cursor()

# DB Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT)')

def add_userdata(username, password):
    c.execute('INSERT INTO userstable(username, password) VALUES (?, ?)', (username, password))
    conn.commit()

def login_user(username, password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, password))
    data = c.fetchall()
    return data

def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data

def generate_synthetic_data(real_data, epochs, weight_clip, sigma, n_critics, batch_size, learning_rate):
    real_data = real_data.fillna('N/A')
    dataset = CategoricalDataset(real_data)
    data_tensor = dataset.to_onehot_flat()

    NOISE_DIM = 100  # Adjust as needed
    HIDDEN_DIM = 64  # Adjust as needed

    gan = create_categorical_gan(NOISE_DIM, HIDDEN_DIM, dataset.dimensions)
    gan.train(data=data_tensor,
              epochs=epochs,
              n_critics=n_critics,
              batch_size=batch_size,
              learning_rate=learning_rate,
              weight_clip=weight_clip,
              sigma=sigma)
    flat_synth_data = gan.generate(len(real_data))
    synth_data = dataset.from_onehot_flat(flat_synth_data)
    return synth_data

def main():
    """Simple Login App"""
    st.title("PrivSynth")

    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'

    # Navigation and Log-out Buttons
    if st.session_state['page'] == 'loggedin':
        # Log-out button on the data input page
        if st.button("Log Out"):
            st.session_state['previous_page'] = st.session_state['page']
            st.session_state['page'] = 'login'
            st.experimental_rerun()

    elif st.session_state['page'] == 'view_data':
        # Back button on the view synthetic data page
        if st.button("Back"):
            st.session_state['page'] = 'loggedin'
            st.experimental_rerun()

    if st.session_state.page == 'login':
        st.subheader("Login Section")

        username = st.text_input("User Name")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            create_usertable()
            hashed_pswd = make_hashes(password)

            if login_user(username, hashed_pswd):
                st.success("Logged In as {}".format(username))
                st.session_state.page = 'loggedin'
            else:
                st.warning("Incorrect Username/Password")

        if st.button("Go to SignUp"):
            st.session_state.page = 'signup'

    elif st.session_state.page == 'signup':
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password", type='password')

        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user, make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.session_state.page = 'login'

        if st.button("Go to Login"):
            st.session_state.page = 'login'

    elif st.session_state.page == 'loggedin':
        st.subheader("Welcome to the dashboard")

        # File uploader
        uploaded_file = st.file_uploader("Upload files", type=["jpeg", "png", "csv", "json", "xlsx"])

        if uploaded_file is not None:
            file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
            st.write(file_details)

            # Additional parameters
            epochs = st.number_input("Number of Epochs", min_value=1, max_value=1000, value=10)
            sigma = st.number_input("Sigma", min_value=0.0, max_value=100.0, value=1.0, step=0.01)
            weight_clip = st.number_input("Weight Clip", min_value=0.0, max_value=100.0, value=1.0, step=0.01)
            n_critics = st.number_input("Number of Critics", min_value=1, value=5)
            batch_size = st.number_input("Batch Size", min_value=1, value=128)
            learning_rate = st.number_input("Learning Rate", min_value=1e-5, max_value=1.0, value=1e-3, format="%.5f")


            st.write("Epochs:", epochs)
            st.write("Sigma:", sigma)
            st.write("Weight Clip:", weight_clip)
            st.write("Number of Critics:", n_critics)
            st.write("Batch Size:", batch_size)
            st.write("Learning Rate:", learning_rate)

            # Process uploaded files based on file type
            if uploaded_file.type == "text/csv" or uploaded_file.type == "application/vnd.ms-excel":
                # Assuming file is a CSV
                df = pd.read_csv(uploaded_file)
                st.table(df.head())
                st.session_state['data'] = df

            elif uploaded_file.type == "application/json":
                # Assuming file is a JSON
                df = pd.read_json(uploaded_file)
                st.table(df.head())
                st.session_state['data'] = df

            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                # Assuming file is an Excel file
                df = pd.read_excel(uploaded_file)
                st.table(df.head())
                st.session_state['data'] = df

            elif "image" in uploaded_file.type:
                # Assuming file is an image
                st.image(uploaded_file.read(), caption="Uploaded Image", use_column_width=True)

            if st.button("Generate Synthetic Data"):
                with st.spinner('Generating synthetic data... Please wait.'):
                # Call the synthetic data generation function
                    try:
                        synth_data = generate_synthetic_data(df, epochs, weight_clip, sigma, n_critics, batch_size, learning_rate)
                        st.session_state['synth_data'] = synth_data
                        st.session_state.page = 'view_data'
                        st.success("Synthetic Data Generated Successfully!")
                    except Exception as e:
                        st.error(f"Error in generating synthetic data: {e}")
                    

    elif st.session_state.page == 'view_data':
        st.subheader("Synthetic Data Visualization")
        if 'synth_data' in st.session_state:
            synth_data = st.session_state['synth_data']
            real_data = st.session_state['data']
            if isinstance(synth_data, pd.DataFrame):
                # Display the synthetic data in a table format
                st.write("Synthetic Data (Tabular):")
                st.table(synth_data.head())

                # Calculate and display dataframe statistics
                st.write("Dataframe Statistics:")
                mi_score = mutual_information_score(real_data, synth_data)
                correlation = correlation_score(real_data, synth_data)
                exact_match = exact_match_score(real_data, synth_data)
                neighbors_privacy = neighbors_privacy_score(real_data, synth_data)
                histogram_similarity = histogram_similarity_dataframes(real_data, synth_data)
                st.write(f"Mutual Information Score: {mi_score:.4f}")
                st.write(f"Correlation Score: {correlation:.4f}")
                st.write(f"Exact Match Score: {exact_match:.4f}")
                st.write(f"Neighbors Privacy Score: {neighbors_privacy:.4f}")
                st.write(f"Histogram Similarity: {histogram_similarity:.4f}")

                # Calculate an overall assessment score (ensemble of metrics)
                assessment_score = (
                    mi_score * 0.2 +  # Adjust weights as needed
                    correlation * 0.2 +
                    exact_match * 0.2 +
                    (1 - neighbors_privacy / 10) * 0.2 +  # Normalize neighbors_privacy to be in [0, 1]
                    histogram_similarity * 0.2
                )

                # Assess the quality of synthetic data based on the overall score
                quality_assessment_text = ""
                if assessment_score >= 0.7:
                    quality_assessment_text = "The synthetic data is of high quality and closely resembles the original data distribution."
                elif 0.4 <= assessment_score < 0.7:
                    quality_assessment_text = "The synthetic data is of moderate quality and reasonably resembles the original data distribution."
                else:
                    quality_assessment_text = "The synthetic data needs improvement to closely match the original data distribution."

                # Assess privacy based on neighbors_privacy score
                privacy_assessment_text = ""
                if neighbors_privacy > 2.0:
                    privacy_assessment_text = "The synthetic data provides reasonable privacy protection."
                else:
                    privacy_assessment_text = "The synthetic data may not provide strong privacy protection. Consider refining the privacy mechanism."

                # Define assessment labels and colors based on scores
                assessment_labels = ["Poor", "Moderate", "High"]
                assessment_colors = ["red", "orange", "green"]
                assessment_index = min(int(assessment_score * len(assessment_labels)), len(assessment_labels) - 1)
                assessment_label = assessment_labels[assessment_index]
                assessment_color = assessment_colors[assessment_index]

                # Define privacy assessment color
                privacy_color = "green" if neighbors_privacy > 2.0 else "red"

                # Display assessments with formatting
                st.write("\nOverall Data Quality Assessment:")
                st.markdown(f"<p style='color:{assessment_color}; font-weight:bold'>{quality_assessment_text}</p>", unsafe_allow_html=True)

                st.write("\nPrivacy Assessment:")
                st.markdown(f"<p style='color:{privacy_color}; font-weight:bold'>{privacy_assessment_text}</p>", unsafe_allow_html=True)

                # Add a button to download synthetic data as a CSV file
                download_button = st.download_button(
                    label="Download Synthetic Data as CSV",
                    data=synth_data.to_csv(index=False).encode(),
                    file_name="synthetic_data.csv",
                    key='download_button'
                )

        elif isinstance(synth_data, list) and all(isinstance(item, Image.Image) for item in synth_data):
            # Display the synthetic images
            st.write("Synthetic Images:")
            for i, image in enumerate(synth_data):
                st.image(image, caption=f"Image {i + 1}", use_column_width=True)

            # Calculate and display image statistics
            st.write("Image Statistics:")
            real_images = []  # Replace with real images
            fid_score = calculate_fid(real_images, synth_data)
            is_mean, is_std = inception_score(synth_data)
            psnr_score = psnr(real_images[0], synth_data[0])  # Example for a single image
            hist_similarity = histogram_similarity_images(real_images[0], synth_data[0])  # Example for a single image
            st.write(f"FID Score: {fid_score:.4f}")
            st.write(f"Inception Score (mean): {is_mean:.4f}")
            st.write(f"Inception Score (std): {is_std:.4f}")
            st.write(f"PSNR Score: {psnr_score:.4f}")
            st.write(f"Histogram Similarity: {hist_similarity:.4f}")

            
if __name__ == '__main__':
    main()
