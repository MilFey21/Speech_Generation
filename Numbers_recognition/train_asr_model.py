import os
# Set environment variable to avoid OpenMP runtime conflicts
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
import multiprocessing

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchaudio
from scipy import signal

torch.set_num_threads(1)
if torch.cuda.is_available():
    torch.cuda.set_device(0)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def augment_audio(y, sr=16000):
    y = y.astype(np.float32)
    
    augmentations = [
        lambda y: librosa.effects.time_stretch(y, rate=np.random.uniform(0.9, 1.1)), 
        lambda y: librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.uniform(-1, 1)), # pitch
        lambda y: y + np.random.normal(0, 0.005, size=y.shape).astype(np.float32), # noise
        lambda y: y * np.random.uniform(0.8, 1.2), # volume
        lambda y: add_simple_reverb(y, sr),
    ]
    
    # we apply random subset of augmentations
    num_augs = np.random.randint(1, 3)  
    selected_augs = np.random.choice(augmentations, num_augs, replace=False)
    
    for aug_func in selected_augs:
        y = aug_func(y)
        y = y.astype(np.float32)
    
    return y.astype(np.float32) 

def add_simple_reverb(y, sr):
    
    delay_time = np.random.uniform(0.1, 0.3)  # seconds
    delay_samples = int(delay_time * sr)
    
    attenuation = np.random.uniform(0.1, 0.5)
    delayed = np.zeros_like(y, dtype=np.float32)
    if delay_samples < len(y):
        delayed[delay_samples:] = y[:-delay_samples] * attenuation
    
    # mix original and delayed signal
    return y + delayed


class NumbersDataset(Dataset):
    
    def __init__(self, csv_file, base_dir, transform=None, cache_dir=None, 
                 sample_rate=16000, max_duration=3.0, augment=False):

        self.data = pd.read_csv(csv_file)
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_length = int(max_duration * sample_rate)
        self.augment = augment
        
        # dataset type from csv filename
        if 'train.csv' in str(csv_file):
            self.dataset_type = 'train'
        elif 'dev.csv' in str(csv_file):
            self.dataset_type = 'dev'
        elif 'test.csv' in str(csv_file):
            self.dataset_type = 'test'
        else:
            self.dataset_type = ''
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loaded {len(self.data)} samples from {csv_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        filename = self.data.iloc[idx]['filename']
        
        if 'transcription' in self.data.columns:
            transcription = self.data.iloc[idx]['transcription']
        else:
            transcription = 0  # Default for test set
        
        # check if filename already contains the dataset type prefix
        if self.dataset_type and filename.startswith(f"{self.dataset_type}/"):
            filename = filename[len(self.dataset_type)+1:]  # +1 for the slash
        
        audio_dir = self.base_dir
        if self.dataset_type and not str(audio_dir).endswith(self.dataset_type):
            audio_dir = audio_dir / self.dataset_type
        possible_paths = [
            audio_dir / filename,
            audio_dir / f"{filename}.wav",
            audio_dir / f"{filename}.mp3",
            audio_dir / Path(filename).with_suffix('.wav'),
            audio_dir / Path(filename).with_suffix('.mp3')
        ]
        
        audio_path = None
        for path in possible_paths:
            if path.exists():
                audio_path = path
                break
        
        if audio_path is None:
            print(f"Warning: Audio file not found: {filename}. Using random features.")
            features = np.random.randn(80, 300).astype(np.float32)
            return torch.tensor(features, dtype=torch.float32), transcription
        
        cache_file = None
        if self.cache_dir:
            cache_file = self.cache_dir / f"{Path(filename).stem}.npy"
            if cache_file.exists() and not self.augment:  # Don't use cache for augmentation
                features = np.load(cache_file).astype(np.float32)
                return torch.tensor(features, dtype=torch.float32), transcription
        
        y, sr = sf.read(str(audio_path))
        # convert to mono
        if len(y.shape) > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            y = y.astype(np.float32)
        
        if self.augment:
            y = augment_audio(y, sr=self.sample_rate)
        
        if len(y) < self.max_length:
            y = np.pad(y, (0, self.max_length - len(y)), 'constant').astype(np.float32)
        else:
            y = y[:self.max_length].astype(np.float32)
        
        # mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=self.sample_rate, n_mels=80, n_fft=512, hop_length=160
        )
        log_mel = librosa.power_to_db(mel_spec)
        
        # normalize
        mean = log_mel.mean()
        std = log_mel.std()
        features = ((log_mel - mean) / (std + 1e-8)).astype(np.float32)
        
        if self.cache_dir and not self.augment:
            np.save(cache_file, features)
        
        return torch.tensor(features, dtype=torch.float32), transcription
    
    def get_filename(self, idx):
        """Get the filename for a specific index."""
        return self.data.iloc[idx]['filename']

# model
class CompactASRModel(nn.Module):
    def __init__(self, num_classes=10, hidden_size=256):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Adding two more convolutional layers
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 6))
        self.dropout1 = nn.Dropout(0.3)
        
        self.flattened_size = 512 * 3 * 6
        self.fc1 = nn.Linear(self.flattened_size, hidden_size)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(0.3)
        
        self.digit_classifiers = nn.ModuleList([
            nn.Linear(hidden_size, num_classes) for _ in range(6)
        ])
        
        self.hidden_size = hidden_size
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)
        
        x = F.relu(self.bn6(self.conv6(x)))
        
        x = self.adaptive_pool(x)
        x = self.dropout1(x)
        x = x.reshape(x.size(0), -1)
        
        if x.size(1) != self.flattened_size:
            print(f"Warning: Expected flattened size {self.flattened_size}, got {x.size(1)}")
            # dynamically adjust the fc1 layer if needed
            old_flattened_size = self.flattened_size
            self.flattened_size = x.size(1)
            new_fc1 = nn.Linear(self.flattened_size, self.hidden_size).to(x.device)
            
            nn.init.normal_(new_fc1.weight, 0, 0.01)
            nn.init.constant_(new_fc1.bias, 0)
            self.fc1 = new_fc1
            
            old_params = old_flattened_size * self.hidden_size + self.hidden_size
            new_params = self.flattened_size * self.hidden_size + self.hidden_size
            print(f"FC1 parameters changed from {old_params:,} to {new_params:,}")
        
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
    
        outputs = []
        for classifier in self.digit_classifiers:
            outputs.append(classifier(x))
        
        return outputs

# lm model
class RussianNumberLM:
    def __init__(self, train_data=None):
        # Initialize with learned probabilities if available
        if train_data is not None:
            self.learn_from_data(train_data)
        else:
            # Default probabilities based on Russian number patterns
            self.position_probs = [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # 1st digit (100,000s)
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # 2nd digit (10,000s)
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # 3rd digit (1,000s)
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # 4th digit (100s)
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # 5th digit (10s)
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # 6th digit (1s)
            ]
        
        # Valid ranges for Russian numbers in the task
        self.min_value = 1000
        self.max_value = 999999
    
        
    def learn_from_data(self, train_data):
        counts = [[0] * 10 for _ in range(6)]
        total = [0] * 6
        
        for num in train_data:
            digits = str(num).zfill(6)
            for pos, digit in enumerate(digits):
                digit = int(digit)
                counts[pos][digit] += 1
                total[pos] += 1
        
        # Convert counts to probabilities
        self.position_probs = []
        for pos in range(6):
            if total[pos] > 0:
                probs = [count/total[pos] for count in counts[pos]]
            else:
                probs = [0.1] * 10
            self.position_probs.append(probs)
    
    def rescore(self, candidates):
        rescored = []
        
        for candidate, score in candidates:
            if self.min_value <= candidate <= self.max_value:
                digits = [int(d) for d in str(candidate).zfill(6)]
                lm_score = 1.0
                
                for pos, digit in enumerate(digits):
                    lm_score *= (self.position_probs[pos][digit] + 0.01)  # Add smoothing
                
                rescored.append((candidate, score * lm_score))
            else:
                # heavily penalize out-of-range numbers
                rescored.append((candidate, score * 0.01))
                
        # sort by new scores
        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored

def beam_search_decode(model, audio_features, lm, beam_width=5):
    model.eval()
    
    if audio_features.dim() == 2:  # [time, freq]
        # Add batch and channel dimensions
        audio_features = audio_features.unsqueeze(0).unsqueeze(0)
    elif audio_features.dim() == 3: 
        audio_features = audio_features.unsqueeze(0)
    
    if next(model.parameters()).device != device:
        print(f"Warning: Model not on {device}, moving it now")
        model = model.to(device)
    audio_features = audio_features.to(device)
    
    with torch.no_grad():
        digit_logits = model(audio_features)
    
    beams = [([], 1.0)]
    
    for pos, logits in enumerate(digit_logits):
        probs = F.softmax(logits.squeeze(0), dim=0).cpu().numpy()
        
        new_beams = []
        for sequence, score in beams:
            for digit in range(10):
                new_seq = sequence + [digit]
                new_score = score * probs[digit]
                new_beams.append((new_seq, new_score))
        
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]
    
    candidates = []
    for sequence, score in beams:
        reversed_seq = sequence[::-1]  # Reverse the sequence
        number = int(''.join(map(str, reversed_seq)))
        
        number = max(1000, min(number, 999999))
        candidates.append((number, score))
    
    rescored = lm.rescore(candidates)
    
    # best candidate
    return rescored[0][0]

def main():
    print("Starting ASR training with full dataset...")
    
    base_dir = "C:/Users/NUC/Documents/ITMO/Speech_Generation/Numbers_recognition/data"
    cache_dir = "C:/Users/NUC/Documents/ITMO/Speech_Generation/Numbers_recognition/cache"
    
    train_dataset = NumbersDataset(
        os.path.join(base_dir, 'train.csv'), 
        base_dir, 
        cache_dir=os.path.join(cache_dir, 'train') if cache_dir else None,
        augment=True  
    )
    
    val_dataset = NumbersDataset(
        os.path.join(base_dir, 'dev.csv'), 
        base_dir,
        cache_dir=os.path.join(cache_dir, 'dev') if cache_dir else None
    )
    
    test_dataset = NumbersDataset(
        os.path.join(base_dir, 'test.csv'), 
        base_dir,
        cache_dir=os.path.join(cache_dir, 'test') if cache_dir else None
    )
    
    num_workers = 2
    
    print(f"Using {num_workers} workers for data loading")
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)  
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    
    model = CompactASRModel().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} parameters")
    print(f"Model is on: {next(model.parameters()).device}")
    
    def digit_loss(outputs, targets):
        batch_size = targets.size(0)
        loss = 0.0
        
        digits = []
        for i in range(6):  # 6 positions
            divisor = 10 ** i
            digit = (targets // divisor) % 10
            digits.append(digit)
        
        # loss for each digit position
        for i, digit_output in enumerate(outputs):
            loss += F.cross_entropy(digit_output, digits[i])
        
        return loss / 6  
    
    def outputs_to_numbers(outputs, batch_size):
        predictions = []
        
        for i in range(batch_size):
            digits = []
            for pos_output in outputs:
                digit_probs = F.softmax(pos_output[i], dim=0)
                digit = torch.argmax(digit_probs).item()
                digits.append(digit)
            
            # digits to number
            number = 0
            for j, digit in enumerate(digits):
                number += digit * (10 ** j)
            number = max(1000, min(number, 999999))
            
            predictions.append(number)
        
        return predictions
    
    def calculate_cer(predictions, targets):
        total_chars = 0
        total_edits = 0
        
        for pred, target in zip(predictions, targets):
            pred_str = str(pred)
            target_str = str(target)
            distance = levenshtein_distance(pred_str, target_str)
            
            total_chars += len(target_str)
            total_edits += distance
        
        return total_edits / total_chars if total_chars > 0 else 1.0

    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    print("Starting training...")
    model.train()
    
    best_val_cer = float('inf')
    num_epochs = 30
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    if torch.cuda.is_available():
        print(f"Initial CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Initial CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        train_loss = 0.0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            if features.dim() == 3:
                features = features.unsqueeze(1)
            
            if epoch == 0 and batch_idx == 0:
                print(f"Input features shape: {features.shape}")
                print(f"Input features device: {features.device}")
                print(f"Labels shape: {labels.shape}")
                print(f"Labels device: {labels.device}")
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = digit_loss(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 300 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Training loss: {avg_train_loss:.4f}")
        
        # validation and CER 
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                if features.dim() == 3:
                    features = features.unsqueeze(1)
                
                outputs = model(features)
                loss = digit_loss(outputs, labels)
                val_loss += loss.item()
                
                batch_predictions = outputs_to_numbers(outputs, features.size(0))
                all_predictions.extend(batch_predictions)
                all_targets.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_cer = calculate_cer(all_predictions, all_targets)
        
        print(f"Validation loss: {avg_val_loss:.4f}, Validation CER: {val_cer:.4f}")
        
        scheduler.step(val_cer)
        
        # best model
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            torch.save(model.state_dict(), "best_asr_model.pt")
            print(f"New best model saved with CER: {best_val_cer:.4f}")
    
    print("Training complete!")
    
    model.load_state_dict(torch.load("best_asr_model.pt"))
    model.eval()
    
    # language model 
    print("Training language model...")
    train_numbers = pd.read_csv(os.path.join(base_dir, 'train.csv'))['transcription'].values
    lm = RussianNumberLM(train_numbers)
    
    # predictions for test 
    print("Generating test predictions...")
    beam_search_predictions = {'filename': [], 'transcription': []}
    direct_predictions = {'filename': [], 'transcription': []}
    
    with torch.no_grad():
        for batch_idx, (features, _) in enumerate(test_loader):
            batch_features = features.to(device)
            if batch_features.dim() == 3:
                batch_features = batch_features.unsqueeze(1)
            
            outputs = model(batch_features)
            batch_direct_preds = outputs_to_numbers(outputs, batch_features.size(0))
            
            for i in range(features.size(0)):
                idx = batch_idx * test_loader.batch_size + i
                if idx < len(test_dataset):
                    filename = test_dataset.get_filename(idx)
                    feature = features[i]
                    if feature.dim() == 2:
                        feature = feature.unsqueeze(0)
                    
                    beam_search_pred = beam_search_decode(model, feature, lm, beam_width=5)
                    
                    direct_pred = batch_direct_preds[i]
                    
                    beam_search_predictions['filename'].append(filename)
                    beam_search_predictions['transcription'].append(beam_search_pred)
                    
                    direct_predictions['filename'].append(filename)
                    direct_predictions['transcription'].append(direct_pred)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Number of CPU cores available: {multiprocessing.cpu_count()}")
    
    main()
    
