import os
# Set environment variable to avoid OpenMP runtime errors
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import random  
import librosa
import warnings
import tqdm
import soundfile as sf
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchaudio
from scipy import signal
from train_asr_model import CompactASRModel, RussianNumberLM, beam_search_decode, augment_audio, add_simple_reverb, NumbersDataset

torch.set_num_threads(1)
if torch.cuda.is_available():
    torch.cuda.set_device(0)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    # Print memory info
    print(f"Total CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")

def test_model(path_to_model, path_to_csv, path_to_audio_dir, output_file="Feiginova_Filin_2.csv"):
    """
    Test the ASR model on the given test set and save predictions.
    
    Args:
        path_to_model: Path to the saved model file (.pt)
        path_to_csv: Path to the test CSV file
        path_to_audio_dir: Path to the directory containing audio files
        output_file: Name of the output CSV file to save predictions
    """
    print(f"Testing model from {path_to_model} on data from {path_to_csv}")
    print(f"Using device: {device}")
    
    cache_dir = os.path.join(os.path.dirname(path_to_model), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    test_dataset = NumbersDataset(
        path_to_csv, 
        path_to_audio_dir,
        cache_dir=os.path.join(cache_dir, 'test')
    )

    pin_memory = torch.cuda.is_available()
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        num_workers=0, pin_memory=pin_memory
    )
    
    train_csv_path = os.path.join(os.path.dirname(path_to_csv), 'train.csv')
    if os.path.exists(train_csv_path):
        print(f"Loading training data for language model from {train_csv_path}")
        train_numbers = pd.read_csv(train_csv_path)['transcription'].values
        lm = RussianNumberLM(train_numbers)
    else:
        print("Training data not found. Using default language model.")
        lm = RussianNumberLM()
    
    test_predictions = {'filename': [], 'transcription': []}

    model = CompactASRModel()
    model.load_state_dict(torch.load(path_to_model))
    model.to(device)
    model.eval()

    print("Starting predictions...")
    with torch.no_grad():
        for batch_idx, (features, _) in enumerate(tqdm.tqdm(test_loader)):
            # process each sample in the batch individually for beam search
            for i in range(features.size(0)):
                feature = features[i]
                if feature.dim() == 2:
                    # add channel dimension
                    feature = feature.unsqueeze(0)
                
                idx = batch_idx * test_loader.batch_size + i
                if idx < len(test_dataset):
                    filename = test_dataset.get_filename(idx)
                    prediction = beam_search_decode(model, feature, lm, beam_width=5)
                    
                    test_predictions['filename'].append(filename)
                    test_predictions['transcription'].append(prediction)
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")

    pd.DataFrame(test_predictions).to_csv(output_file, index=False)
    print(f"Test predictions saved to {output_file}")


def main():
    """
    Main function to parse command line arguments and run the test.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ASR model on Russian numbers.')
    parser.add_argument('--model', type=str, default="best_asr_model.pt",
                        help='Path to the saved model file (.pt)')
    parser.add_argument('--csv', type=str, default="data/test.csv",
                        help='Path to the test CSV file')
    parser.add_argument('--audio_dir', type=str, default="data",
                        help='Path to the directory containing audio files')
    parser.add_argument('--output', type=str, default="Feiginova_Filin_2.csv",
                        help='Name of the output CSV file to save predictions')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output for diagnosis')
    
    args = parser.parse_args()
    
    if args.debug:
        print("Debug mode enabled - printing additional information")
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")
        
        if torch.cuda.is_available():
            print(f"Number of CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"Device {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    test_model(args.model, args.csv, args.audio_dir, args.output)


if __name__ == "__main__":
    main()
