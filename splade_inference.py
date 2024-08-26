import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from qdrant_client import QdrantClient, models
import os
from tqdm import tqdm


class SpladeEncoder(torch.nn.Module):
    def __init__(self, model_name: str, token: str, device: torch.device):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, token=token).to(
            self.device
        )
        self.model = torch.compile(self.model)
        self.model.eval()
        self.threshold = 0.3

    @torch.compile()
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return torch.log1p(torch.relu(logits))

    @torch.inference_mode()
    def encode_batch(self, batch_texts):
        tokens = self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        activated = self.forward(input_ids, attention_mask)

        # Use torch.where and keep everything on GPU
        masked_activated = torch.where(
            attention_mask.unsqueeze(-1).bool(),
            activated,
            torch.tensor(-float("inf"), device=self.device),
        )

        doc_reps, _ = torch.max(masked_activated, dim=1)

        # Apply threshold on GPU
        doc_reps = torch.where(
            doc_reps > self.threshold, doc_reps, torch.zeros_like(doc_reps)
        )

        # Move to CPU only at the end
        indices = [torch.nonzero(doc_rep).squeeze(1).cpu() for doc_rep in doc_reps]
        values = [doc_rep[idx].cpu() for doc_rep, idx in zip(doc_reps, indices)]
        return indices, values


def read_topics(file_path):
    topics = []
    with open(file_path, "r") as f:
        for line in f:
            topic_id, query = line.strip().split("\t")
            topics.append((topic_id, query))
    return topics


def search_and_write_results(client, encoder, topics, output_file, top_k=125):
    with open(output_file, "w") as f:
        for topic_id, query in tqdm(topics):
            indices, values = encoder.encode_batch([query])

            result = client.search(
                collection_name="spladev3_thresh_0_3",
                query_vector=models.NamedSparseVector(
                    name="text",
                    vector=models.SparseVector(
                        indices=indices[0].tolist(),
                        values=values[0].tolist(),
                    ),
                ),
                limit=top_k,
                with_payload=True,
                with_vectors=True,
            )

            for rank, hit in enumerate(result, 1):
                score = hit.score
                doc_id = hit.payload["docid"]  # Correctly extract docid from payload
                f.write(f"{topic_id} Q0 {doc_id} {rank} {score:.8f} splade_v3\n")


def main():
    # Initialize Qdrant client
    client = QdrantClient("localhost", port=6333)

    # Initialize SpladeEncoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "naver/splade-v3"
    token = "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"
    encoder = SpladeEncoder(model_name, token, device)

    # Read topics
    topics_file = "topics.rag24.raggy-dev.txt"
    topics = read_topics(topics_file)

    # Search and write results
    output_file = "raggy-dev_results.txt"
    search_and_write_results(client, encoder, topics, output_file, top_k=200)

    print(f"Search results have been written to {output_file}")


if __name__ == "__main__":
    main()
