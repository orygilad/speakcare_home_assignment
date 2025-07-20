from torch.nn.functional import cosine_similarity, softmax
import torch
import config
def predict_speaker_per_test_segment(test_embeddings , reference_embeddings , speakers , speaker_to_indices):
    certainty_values = []
    predicted_speaker_values = []
    all_speaker_probs = []
    filtered_probs = []
    for i, test_emb in enumerate(test_embeddings):
        test_tensor = test_emb.unsqueeze(0)  # [1, D]
        similarities = cosine_similarity(test_tensor, reference_embeddings, dim=1)  # [N]

        # Aggregate similarities per speaker (max)
        speaker_logits = []
        for speaker in speakers:
            segment_idxs = speaker_to_indices[speaker]
            max_score = similarities[segment_idxs].max().item()
            speaker_logits.append(max_score)

        speaker_logits = torch.tensor(speaker_logits)
        probabilities = softmax(speaker_logits, dim=0)  # [num_speakers]

        best_idx = torch.argmax(probabilities).item()
        predicted_speaker = speakers[best_idx]
        certainty = probabilities[best_idx].item()

        all_speaker_probs.append(probabilities.detach().cpu())

        if certainty < config.CERTAINTY_THRESHOLD:
            predicted_speaker = "unknown"
        else:
            certainty_values.append(certainty)
            predicted_speaker_values.append(predicted_speaker)
            filtered_probs.append(probabilities.detach().cpu())  # ✅ only keep valid ones

        print(f"🧠 Segment {i+1:02d}: Speaker = {predicted_speaker:15s} | Certainty = {certainty:.3f}")
    return all_speaker_probs,filtered_probs , predicted_speaker_values , certainty_values
def get_speaker_name(label: str) -> str:
    return label.split("_seg")[0].replace(".wav", "")