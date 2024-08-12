import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Dict, Tuple
import time
import datasets
from qdrant_client import QdrantClient, models, AsyncQdrantClient
import asyncio
from qdrant_client.http.models import PointStruct, SparseVector
import re
import uuid
import logging
from tqdm import tqdm
import os

torch.set_float32_matmul_precision("high")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="11_8_1800.log",
    filemode="w",
)
logging.getLogger("httpx").setLevel(logging.WARNING)


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


async def batch_insert_splade_embeddings(
    client: AsyncQdrantClient,
    collection_name: str,
    doc_ids: List[str],
    all_indices: List[List[int]],
    all_values: List[List[float]],
):
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            payload={"docid": doc_id},
            vector={
                "text": models.SparseVector(
                    indices=indices.tolist(),
                    values=values.tolist(),
                )
            },
        )
        for doc_id, indices, values in zip(doc_ids, all_indices, all_values)
    ]

    await client.upsert(collection_name=collection_name, points=points, wait=False)


async def process_and_insert_embeddings(
    dataset, encoder, client, collection_name, batch_size=64
):
    start_time = time.time()
    processed_count = 0
    error_count = 0
    error_doc_ids = []

    for batch in tqdm(dataset.iter(batch_size=batch_size), desc="Processing batches"):
        try:
            batch_texts = batch["segment"]
            batch_ids = batch["docid"]

            all_indices, all_values = encoder.encode_batch(batch_texts)

            await batch_insert_splade_embeddings(
                client, collection_name, batch_ids, all_indices, all_values
            )

            processed_count += len(batch_texts)
            logging.info(
                f"Processed and initiated upsert for {processed_count} documents"
            )

        except Exception as e:
            error_count += 1
            error_doc_ids.extend(batch_ids)
            logging.error(f"Error processing batch: {str(e)}")
            logging.error(f"Error doc ids: {error_doc_ids}")
    if error_count > 0:
        with open("error_doc_ids.txt", "w") as f:
            for doc_id in error_doc_ids:
                f.write(f"{doc_id}\n")
    end_time = time.time()
    print(
        f"Processed and initiated upserts for {processed_count} documents in {end_time - start_time:.2f} seconds"
    )


async def check_collection_exists(client: AsyncQdrantClient, collection_name: str):
    if await client.collection_exists(collection_name):
        await client.delete_collection(collection_name)


async def create_collection(client: AsyncQdrantClient, collection_name: str):
    await client.create_collection(
        collection_name=collection_name,
        vectors_config={},
        sparse_vectors_config={
            "text": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=True)
            )
        },
    )


async def main():
    try:
        access_token = os.environ["HF_ACCESS_TOKEN"]
        if not access_token:
            raise ValueError("HF_ACCESS_TOKEN environment variable is not set.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = SpladeEncoder("naver/splade-v3", access_token, device)
        dataset = datasets.load_dataset(
            "/root/data/msmarco_v2.1_doc_segmented/", streaming=True, split="train"
        )

        # Initialize Qdrant client
        client = AsyncQdrantClient(
            "localhost", port=6333, grpc_port=6334, prefer_grpc=True
        )

        # Create or recreate the collection
        collection_name = "spladev3_thresh_0_3"

        await check_collection_exists(client, collection_name)
        await create_collection(client, collection_name)

        # Process and insert embeddings
        await process_and_insert_embeddings(dataset, encoder, client, collection_name)

        logging.info("Indexing completed.")
    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
