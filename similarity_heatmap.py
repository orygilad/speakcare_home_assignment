import os
import torch
import torchaudio
import numpy as np
from feature_extractor import FeatureExtractor
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import config
import audio_utils
extractor = FeatureExtractor("models/speechbrain_ecapa")

segment_embeddings = []
segment_labels = []

print(f"Scanning {config.DATA_DIR} for speaker audio...")

for speaker_file in os.listdir(config.DATA_DIR):
    if not speaker_file.endswith(".wav"):
        continue
    signal, fs = audio_utils.load_audio(os.path.join(config.DATA_DIR, speaker_file))
    total_samples = signal.shape[1]
    segment_len = config.SEGMENT_SECONDS * config.SAMPLE_RATE
    num_segments = total_samples // segment_len

    print(f"{speaker_file}: splitting into {num_segments} segments...")

    for i in range(num_segments):
        start = i * segment_len
        end = start + segment_len
        segment = signal[:, start:end]

        with torch.no_grad():
            embedding = extractor.extract_embedding_from_tensor(segment, config.SAMPLE_RATE)
        #segment_embeddings.append(torch.tensor(embedding))
        segment_labels.append(f"{speaker_file}_seg{i+1}")
    segment_embeddings.extend(extractor.segment_and_embed(signal))
print("Extracted embeddings. Computing cosine similarity matrix...")

embeddings_tensor = torch.stack(segment_embeddings)
similarity_matrix = cosine_similarity(
    embeddings_tensor.unsqueeze(1),  # shape: (N, 1, D)
    embeddings_tensor.unsqueeze(0),  # shape: (1, N, D)
    dim=-1
)  # shape: (N, N)
# Convert to numpy for plotting
sim_matrix_np = similarity_matrix.numpy()
# Normalize cosine similarity to [0, 1] range
norm_sim_matrix = (sim_matrix_np + 1) / 2  # from [-1, 1] -> [0, 1]

# Add small epsilon to avoid log(0)
epsilon = 1e-5
log_sim_matrix = 10*np.log10(norm_sim_matrix + epsilon)
# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(log_sim_matrix, xticklabels=segment_labels, yticklabels=segment_labels,
            cmap="viridis", square=True, cbar_kws={"label": "Cosine Similarity[dB]"})
plt.title("Cosine Similarity Between Segments")
plt.xticks(rotation=90, fontsize=6)
plt.yticks(rotation=0, fontsize=6)
plt.tight_layout()
plt.show()
