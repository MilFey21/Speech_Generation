from typing import List, Tuple

import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import heapq


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-960h",
            lm_model_path="3-gram.pruned.1e-7.arpa",
            beam_width=3,
            alpha=1.0,
            beta=1.0
        ):
        """
        Initialization of Wav2Vec2Decoder class
        
        Args:
            model_name (str): Pretrained Wav2Vec2 model from transformers
            lm_model_path (str): Path to the KenLM n-gram model (for LM rescoring)
            beam_width (int): Number of hypotheses to keep in beam search
            alpha (float): LM weight for shallow fusion and rescoring
            beta (float): Word bonus for shallow fusion
        """
        # once logits are available, no other interactions with the model are allowed
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        # you can interact with these parameters
        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path)
        
        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V)
        
        Returns:
            str: Decoded transcript
        """
        # Get most probable token indices at each time step and filter out
        indices = torch.argmax(logits, dim=-1)  # Shape: (T,)
        indices = torch.unique_consecutive(indices, dim=-1)
        filtered_indices = [idx.item() for idx in indices if idx != self.blank_token_id]
        
        # Map token indices to characters
        transcript_chars = [self.vocab[idx] for idx in filtered_indices]
        transcript = "".join(transcript_chars)
        
        # (?) Replace word delimiter token with space (if defined)
        if self.word_delimiter is not None:
            transcript = transcript.replace(self.word_delimiter, " ")
        
        return transcript
        
    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM)
        
        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size
            return_beams (bool): Return all beam hypotheses for second pass LM rescoring
        
        Returns:
            Union[str, List[Tuple[float, List[int]]]]: 
                (str) - If return_beams is False, returns the best decoded transcript as a string.
                (List[Tuple[List[int], float]]) - If return_beams is True, returns a list of tuples
                    containing hypotheses and log probabilities.
        """
        T, V = logits.size()
        log_probs = torch.log_softmax(logits, dim=-1) 

        # beams: log_prob, sequence
        beams = [(0.0, [])]

        for t in range(T):
            new_beams = []
            for log_prob, seq in beams:
                # all possible next tokens
                for v in range(V):
                    new_seq = seq + [v]
                    new_log_prob = log_prob + log_probs[t, v].item()
                    new_beams.append((new_log_prob, new_seq))
            
            # top-k beams 
            beams = heapq.nlargest(self.beam_width, new_beams, key=lambda x: x[0])

        if return_beams:
            # processed beams with log probabilities
            processed_beams = []
            for log_prob, hyp in beams:
                processed_tokens = self._process_ctc(hyp)
                processed_beams.append((processed_tokens, log_prob))
                #processed_beams.append((log_prob, processed_tokens))
            return processed_beams
        else:
            # best hypothesis 
            best_hyp = beams[0][1] if beams else []
            processed_tokens = self._process_ctc(best_hyp)
            return self._tokens_to_text(processed_tokens)


    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding with shallow LM fusion
        
        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size
        
        Returns:
            str: Decoded transcript
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")
        
        T, V = logits.size()
        log_probs = torch.log_softmax(logits, dim=-1)  # (T, V)
        
        # beams: total_score, acoustic_score, lm_score, processed_tokens
        beams = [(0.0, 0.0, 0.0, [])]
        
        for t in range(T):
            new_beams = []
            for beam in beams:
                total_score, acoustic_score, lm_score, processed = beam
                
                for v in range(V):
                    new_acoustic = acoustic_score + log_probs[t, v].item()
                    new_sequence = processed + [v]
                    # collapse duplicates, remove blanks
                    processed_sequence = self._process_ctc(new_sequence)
                    # calculate LM score and word bonus
                    text = self._tokens_to_text(processed_sequence)
                    words = text.split()
                    current_lm_score = self.lm_model.score(" ".join(words), bos=True, eos=False)
                    word_bonus = self.beta * len(words)
                    
                    # combine scores
                    new_total = new_acoustic + self.alpha * current_lm_score + word_bonus
                    new_beams.append((new_total, new_acoustic, current_lm_score, new_sequence))
            
            # top-k beams
            beams = sorted(new_beams, key=lambda x: -x[0])[:self.beam_width]
        
        best_beam = max(beams, key=lambda x: x[0], default=(0, 0, 0, []))
        return self._tokens_to_text(self._process_ctc(best_beam[3]))

    def _process_ctc(self, indices: List[int]) -> List[int]:
        """Collapse repeats and remove blank tokens"""
        processed = []
        prev = None
        for idx in indices:
            if idx == self.blank_token_id:
                continue
            if idx != prev:
                processed.append(idx)
                prev = idx
        return processed

    def _tokens_to_text(self, tokens: List[int]) -> str:
        """Convert token IDs to text with word boundary handling"""
        text = "".join([self.vocab.get(idx, "") for idx in tokens])
        if self.word_delimiter:
            text = text.replace(self.word_delimiter, " ")
        return text.strip()


    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs
        
        Args:
            beams (list): List of tuples (hypothesis, log_prob)
        
        Returns:
            str: Best rescored transcript
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")
        
        best_score = -float('inf')
        best_text = ""
        
        for hyp_tokens, log_prob in beams:
            processed_tokens = self._process_ctc(hyp_tokens)
            text = self._tokens_to_text(processed_tokens)
            
            # calculate LM score 
            lm_score = self.lm_model.score(text, bos=True, eos=False)
            words = text.split()
            word_count = len(words)
            
            total_score = (
                log_prob +  # acoustic score
                self.alpha * lm_score +  # LM score
                self.beta * word_count  # Word insertion bonus
            )
            
            # best hypothesis
            if total_score > best_score:
                best_score = total_score
                best_text = text
        
        return best_text.strip()

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Decode input audio file using the specified method
        
        Args:
            audio_input (torch.Tensor): Audio tensor
            method (str): Decoding method ("greedy", "beam", "beam_lm", "beam_lm_rescore"),
                where "greedy" is a greedy decoding,
                      "beam" is beam search without LM,
                      "beam_lm" is beam search with LM shallow fusion, and 
                      "beam_lm_rescore" is a beam search with second pass LM rescoring
        
        Returns:
            str: Decoded transcription
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError("Invalid decoding method. Choose one of 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'.")


def test(decoder, audio_path, true_transcription):

    import Levenshtein

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Audio sample rate must be 16kHz"

    print("=" * 60)
    print("Target transcription")
    print(true_transcription)

    def calculate_wer(ref, hyp):
        """Calculate Word Error Rate (WER)"""
        ref_words = ref.split()
        hyp_words = hyp.split()
        
        # Levenshtein distance at the word level
        d = Levenshtein.distance(ref_words, hyp_words)
        wer = float(d) / len(ref_words) if ref_words else 0.0
        return wer

    # Print all decoding methods results
    for d_strategy in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        print("-" * 60)
        print(f"{d_strategy} decoding") 
        transcript = decoder.decode(audio_input, method=d_strategy)
        print(f"{transcript}")
        print(f"Character-level Levenshtein distance: {Levenshtein.distance(true_transcription, transcript.strip())}")
        wer = calculate_wer(true_transcription, transcript.strip()) 
        print(f"Word Error Rate (WER): {wer:.4f}")


if __name__ == "__main__":
    
    test_samples = [
        ("examples/sample1.wav", "IF YOU ARE GENEROUS HERE IS A FITTING OPPORTUNITY FOR THE EXERCISE OF YOUR MAGNANIMITY IF YOU ARE PROUD HERE AM I YOUR RIVAL READY TO ACKNOWLEDGE MYSELF YOUR DEBTOR FOR AN ACT OF THE MOST NOBLE FORBEARANCE"),
        ("examples/sample2.wav", "AND IF ANY OF THE OTHER COPS HAD PRIVATE RACKETS OF THEIR OWN IZZY WAS UNDOUBTEDLY THE MAN TO FIND IT OUT AND USE THE INFORMATION WITH A BEAT SUCH AS THAT EVEN GOING HALVES AND WITH ALL THE GRAFT TO THE UPPER BRACKETS HE'D STILL BE ABLE TO MAKE HIS PILE IN A MATTER OF MONTHS"),
        ("examples/sample3.wav", "GUESS A MAN GETS USED TO ANYTHING HELL MAYBE I CAN HIRE SOME BUMS TO SIT AROUND AND WHOOP IT UP WHEN THE SHIPS COME IN AND BILL THIS AS A REAL OLD MARTIAN DEN OF SIN"),
        ("examples/sample4.wav", "IT WAS A TUNE THEY HAD ALL HEARD HUNDREDS OF TIMES SO THERE WAS NO DIFFICULTY IN TURNING OUT A PASSABLE IMITATION OF IT TO THE IMPROVISED STRAINS OF I DIDN'T WANT TO DO IT THE PRISONER STRODE FORTH TO FREEDOM"),
        ("examples/sample5.wav", "MARGUERITE TIRED OUT WITH THIS LONG CONFESSION THREW HERSELF BACK ON THE SOFA AND TO STIFLE A SLIGHT COUGH PUT UP HER HANDKERCHIEF TO HER LIPS AND FROM THAT TO HER EYES"),
        ("examples/sample6.wav", "AT THIS TIME ALL PARTICIPANTS ARE IN A LISTEN ONLY MODE"),
        ("examples/sample7.wav", "THE INCREASE WAS MAINLY ATTRIBUTABLE TO THE NET INCREASE IN THE AVERAGE SIZE OF OUR FLEETS"),
        ("examples/sample8.wav", "OPERATING SURPLUS IS A NON CAP FINANCIAL MEASURE WHICH IS DEFINED AS FULLY IN OUR PRESS RELEASE"),
    ]

    decoder = Wav2Vec2Decoder()

    _ = [test(decoder, audio_path, target) for audio_path, target in test_samples]
