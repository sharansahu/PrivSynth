from scipy import linalg
import numpy as np
import torch
from torchvision.models import inception_v3
from torchvision.transforms import functional as TF
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors

# Synthetic Data Statistics for Images

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Compute the Inception Score of generated images.
    
    Args:
        imgs (List of PIL Images): Images to be evaluated.
        cuda (bool): Whether to use CUDA (GPU support).
        batch_size (int): Batch size for processing.
        resize (bool): Resize images to (299, 299) as required by Inception model.
        splits (int): Number of splits for calculating the score.

    Returns:
        Tuple of (mean, std) of the Inception Score.
    """
    N = len(imgs)
    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = torch.nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return torch.nn.functional.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = torch.autograd.Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def calculate_fid(images1, images2, batch_size=50, dims=2048, device='cuda', resize=False):
    """Calculate the FID score between two sets of images.
    
    Args:
        images1 (List of PIL Images): First set of images.
        images2 (List of PIL Images): Second set of images.
        batch_size (int): Batch size for processing.
        dims (int): Dimensionality of features extracted from Inception model.
        device (str): Device to run the computation ('cuda' or 'cpu').
        resize (bool): Resize images to (299, 299) for Inception model.

    Returns:
        float: FID score.
    """
    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = torch.nn.Identity()
    inception_model = inception_model.to(device)
    inception_model.eval()

    def get_activations(images, model, batch_size, dims, device, resize):
        """Calculates the activations of the Inception model for a set of images"""
        num_images = len(images)
        pred_arr = np.empty((num_images, dims))
        for i in range(0, num_images, batch_size):
            batch_imgs = images[i:i + batch_size]
            if resize:
                batch_imgs = [img.resize((299, 299)) for img in batch_imgs]
            batch = torch.stack([TF.to_tensor(s) for s in batch_imgs]).to(device)
            with torch.no_grad():
                pred = model(batch)
            pred_arr[i:i + batch_size] = pred.cpu().numpy().reshape(batch.size(0), -1)
        return pred_arr

    # Get activations for both sets of images
    act1 = get_activations(images1, inception_model, batch_size, dims, device, resize)
    act2 = get_activations(images2, inception_model, batch_size, dims, device, resize)

    # Compute FID
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def psnr(original, compressed):
    """Compute the Peak Signal to Noise Ratio between two images.
    
    Args:
        original (numpy.ndarray): Original image.
        compressed (numpy.ndarray): Compressed or generated image.

    Returns:
        float: PSNR value.
    """
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal.
                  # Therefore PSNR is infinite (perfect similarity).
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Synthetic data statistics for images and dataframes

def histogram_similarity_images(images1, images2, bins=64):
    # Compute histograms
    hist1 = np.histogram(images1, bins=bins, range=(0, 255))[0]
    hist2 = np.histogram(images2, bins=bins, range=(0, 255))[0]

    # Calculate similarity (e.g., using cosine similarity)
    similarity = np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2))
    return similarity

def histogram_similarity_dataframes(df1, df2, bins=64):
    # Select only numeric columns
    numeric_columns_df1 = df1.select_dtypes(include=[np.number])
    numeric_columns_df2 = df2.select_dtypes(include=[np.number])

    # Check if there are numeric columns to compare
    if numeric_columns_df1.empty or numeric_columns_df2.empty:
        return 0.0  # Return 0 if no numeric columns are available

    # Compute histograms for numeric columns
    hist1 = np.histogram(numeric_columns_df1, bins=bins, range=(0, 255))
    hist2 = np.histogram(numeric_columns_df2, bins=bins, range=(0, 255))

    # Calculate similarity (e.g., using cosine similarity)
    similarity = np.dot(hist1[0], hist2[0]) / (np.linalg.norm(hist1[0]) * np.linalg.norm(hist2[0]))
    return similarity

# Synthetic data statistics for dataframes

def mutual_information_score(df1, df2):
    scores = []
    # Select only numeric columns
    numeric_columns_df1 = df1.select_dtypes(include=[np.number])
    numeric_columns_df2 = df2.select_dtypes(include=[np.number])
    for col in numeric_columns_df1.columns:
        score = mutual_info_score(numeric_columns_df1[col], numeric_columns_df2[col])
        scores.append(score)
    return np.mean(scores)

def correlation_score(df1, df2):
    # Select only numeric columns
    numeric_columns_df1 = df1.select_dtypes(include=[np.number])
    numeric_columns_df2 = df2.select_dtypes(include=[np.number])
    corr1 = numeric_columns_df1.corr().values.flatten()
    corr2 = numeric_columns_df2.corr().values.flatten()
    return np.corrcoef(corr1, corr2)[0, 1]

def exact_match_score(df1, df2):
    merged_df = df1.merge(df2, indicator=True, how='outer')
    exact_matches = merged_df[merged_df['_merge'] == 'both'].shape[0]
    return exact_matches / min(len(df1), len(df2))

def neighbors_privacy_score(df1, df2, n_neighbors=5):
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    # Select only numeric columns
    numeric_columns_df1 = df1.select_dtypes(include=[np.number])
    nn.fit(numeric_columns_df1)
    numeric_columns_df2 = df2.select_dtypes(include=[np.number])
    distances, _ = nn.kneighbors(numeric_columns_df2)
    return distances.mean()
