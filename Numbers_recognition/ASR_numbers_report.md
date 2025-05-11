## ASR - Numbers Recognition in Russian
**Team:** Milena Feiginova/Vasilii Filin

**Competition Overview:**

- Task: Transcribe spoken Russian numbers (1,000-999,999)
- Constraints: Model size ≤ 5M parameters
- Audio: 16 kHz (mp3 and wav formats)
- Evaluation: Harmonic mean of CER on in-domain and out-of-domain voices
- Submission: CSV with filename and integer transcription



#### Data Pipeline

Our pipeline consists of the following components:

1. **Data Loading**: The `NumbersDataset` class handles loading audio files in various formats (`.wav` and `.mp3`), with intelligent path resolution.

2. **Audio Preprocessing**:
   - Conversion to 16kHz mono audio
   - Padding/trimming to a standard 3-second length
   - Extraction of 80-dimensional Mel spectrogram features
   - Per-sample normalization (mean=0, std=1)

3. **Data Augmentation**:
   - Time stretching (0.9-1.1x)
   - Pitch shifting (±1 semitone)
   - Adding Gaussian noise
   - Volume adjustment (0.8-1.2x)
   - Simulated reverb effect

4. **Feature Caching**: To accelerate training, processed features are cached to disk once computed.

#### Model Architecture

The `CompactASRModel` architecture is designed specifically for the Russian numbers recognition task:

Key features:
- 6 convolutional layers with batch normalization and max pooling (128→128→256→256→512→256)
- Adaptive pooling to handle variable input lengths
- Dropout for regularization (0.3-0.4)
- Separate classifier for each digit position (6 digit positions total)
- Model has 4,658,364 trainable parameters

#### Training Strategy


1. **Loss Function**: Average cross-entropy loss across all 6 digit positions
2. **Optimizer**: AdamW with initial learning rate of 0.001
3. **Learning Rate Scheduler**: ReduceLROnPlateau with:
   - Monitoring validation Character Error Rate (CER)
   - Reducing learning rate by factor of 0.5 when improvements plateau
   - Patience of 3 epochs

4. **Batch Size**: 32 samples
5. **Epochs**: After experimentation, we found 20 epochs to be optimal (vs. the initial 30)
   - With longer training, the model began to overfit
   - The scheduler typically triggered 2-3 learning rate reductions during training

#### Language Model & Beam Search Decoding

To improve accuracy, we implemented:

1. **Russian Number Language Model (`RussianNumberLM`):**
   - Learns digit probability distributions from training data
   - Enforces valid range constraints (1,000-999,999)
   - Provides score adjustments for candidate predictions

2. **Beam Search Decoding:**
   - Uses beam width of 5 for considering multiple hypothesis paths
   - Scores candidates based on both acoustic model confidence and language model probability
   - Rescores candidates to strongly prefer valid number ranges

The beam search approach significantly improved accuracy compared to direct argmax decoding, reducing Character Error Rate (CER) by approximately 15-20%.

#### Experiment Results

Our experiments revealed several important insights:

1. **Optimal Training Duration**: 
   - 20 epochs provided the best balance of accuracy vs. overfitting
   - Longer training (30+ epochs) showed diminishing returns and sometimes performance degradation

2. **Scheduler Effectiveness**:
   - The ReduceLROnPlateau scheduler was crucial for reaching optimal performance
   - Typically, the learning rate would be reduced 2-3 times during training
   - Each reduction generally improved validation metrics by 3-5%

3. **Beam Search Impact**:
   - Beam search with language model rescoring reduced error rates by 15-20% compared to greedy decoding
   - Wider beam widths (>5) showed diminishing returns for the additional computation cost

4. **Data Augmentation**:
   - Random augmentations improved robustness and reduced overfitting
   - Time stretching and pitch shifting provided the most benefit

#### History of submissions 
1. Baseline model with 1m parameters, no data augmentation. **Score:** 89
2. Added data augmentation. **Score:** 90
3. Added beam search decoding but found a mistake in it later. **Score:** 86
4. Increased model size to 2m parameters. **Score:** 65
5. Fixed beam search decoding. **Score:** 34
6. Added language model. **Score:** 31
7. Increased model size, but later found out that it exceeded the restriction of 5 million parameters **Score:** 18
8. Tested inference pipeline
9. Decreased model size to 4.6 million parameters **Score:** 18.641

#### Conclusion

Our solution to the Russian Numbers ASR challenge combines a specially designed CNN architecture with effective data augmentation and beam search decoding. The experimental results demonstrate that a 20-epoch training regimen with learning rate scheduling and beam search decoding provides the optimal balance of accuracy and computational efficiency.

The integration of a language model through beam search decoding proves particularly effective for this task, as spoken numbers follow predictable patterns that can be leveraged to improve recognition accuracy.

Future improvements could explore transformer-based architectures or more sophisticated data augmentation techniques specific to number recognition tasks.
