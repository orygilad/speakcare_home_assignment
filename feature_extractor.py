from speechbrain.pretrained import SpeakerRecognition
import torch
from pathlib import Path
import os
import config

class FeatureExtractor:
    def __init__(self, model_dir, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model from local directory without symlinks
        self.model = SpeakerRecognition.from_hparams(
            source=model_dir,
            savedir=os.path.join(model_dir, "local_copy"),
            run_opts={"device": self.device}
        )

    def extract_embedding(self, wav_path):
        import torchaudio
        wav_path = str(Path(wav_path).resolve())
        signal, fs = torchaudio.load(wav_path)
        with torch.no_grad():
            embedding = self.model.encode_batch(signal.to(self.device))
        return embedding.squeeze().cpu().numpy()
    def extract_embedding_from_tensor(self, signal_tensor, sample_rate):
        # Expected format: (1, num_samples)
        return self.model.encode_batch(signal_tensor).squeeze().cpu().numpy()
    def segment_and_embed(self,signal):
        total_samples = signal.shape[1]
        segment_len = config.SEGMENT_SECONDS * config.SAMPLE_RATE
        num_segments = total_samples // segment_len
        segment_embeddings=[]
        for i in range(num_segments):
            start = i * segment_len
            end = start + segment_len
            segment = signal[:, start:end]

            with torch.no_grad():
                embedding = self.extract_embedding_from_tensor(segment, config.SAMPLE_RATE)

            segment_embeddings.append(torch.tensor(embedding))
        return segment_embeddings
