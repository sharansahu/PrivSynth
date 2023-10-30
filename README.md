# PrivSynth: A Streamlit App for Generating Differentially Private Tabular Data

PrivSynth is a Streamlit application designed to create differentially private tabular data using Differentially Private Wasserstein Generative Adversarial Networks (DPWGAN). This tool is particularly useful for users who need to generate synthetic data sets that closely resemble original data while ensuring the privacy of individual data entries.

## Key Features
1. **Ease of Use**: The Streamlit interface makes it straightforward to set up and use, even for those with limited technical background in data privacy or machine learning.
2. **Privacy Preservation**: Incorporates differential privacy techniques to protect individual data entries in the synthetic data.
3. **Customizable GAN Models**: Allows users to define and tweak the generator and discriminator models based on their specific data and requirements.
4. **Versatile Application**: Suitable for various types of tabular data, especially where privacy is a paramount concern, such as in medical or financial datasets.
5. **User Authentication**: Features a secure login and signup system using SQLite, ensuring that only authorized users can access the tool and create synthetic datasets.
6. **Statistical Insights**: Displays statistics for privacy and distribution match, providing users with clear indicators of whether the synthetic data maintains privacy and closely matches the original dataset.
7. **Parameter Customization**: Enables users to define and adjust critical parameters such as epochs, sigma, weight clip, number of critics, batch size, and learning rate. These settings give users control over the level of differential privacy and the quality of the synthetic data generated.


## Installation
To install PrivSynth, simply run:

```
pip install -r requirements.txt
```

## Usage
After installation, you can start the PrivSynth app locally using:

```
streamlit run app.py
```

The app provides a user-friendly interface for setting up the DPWGAN, including the definition of generator and discriminator models, as well as configuring the noise function and other training parameters.

## Model

This model is largely an implementation of the [Differentially Private Generative Adversarial Network model](https://arxiv.org/abs/1802.06739)
from Xie, Lin, Wang, Wang, and Zhou (2018).

### Setting Up the DPWGAN
1. **Define the Generator and Discriminator Models**: PrivSynth allows for the customization of these models. Users can select different layers, activation functions, and other architectural details.
   
2. **Configure the Noise Function**: The noise function is crucial in generating the initial input for the generator model. This function can be tailored according to the nature of the data being synthesized.

### Training and Generating Data
- **Train the DPWGAN**: Once the setup is complete, users can train the DPWGAN using their own data. PrivSynth provides options to adjust training parameters such as batch size, number of epochs, and learning rates.
- **Generate Synthetic Data**: After training, users can generate synthetic data that mirrors the statistical properties of the original dataset while maintaining privacy.

## Privacy Calculations
PrivSynth integrates methods to calculate the privacy loss (epsilon) for the synthetic data generation process, adhering to the principles of differential privacy. This calculation is crucial for understanding the trade-off between data utility and privacy.

## Examples and Documentation
PrivSynth comes with example scripts and detailed documentation to guide users through the process of generating differentially private synthetic data. These resources are invaluable for understanding the best practices and nuances of working with DPWGANs.

## Acknowledgements
PrivSynth builds upon the foundational work of DPWGAN, initially developed and provided by the team at Civis Analytics. The tool enhances their original implementation, making it more accessible and applicable for a broader range of users and use cases.

---

PrivSynth stands out as a practical, accessible solution for generating differentially private synthetic data, catering to the growing need for privacy-preserving data analysis tools in various industries.
