from qdrant_client import QdrantClient

if __name__ == "__main__":
    client = QdrantClient("localhost", port=6333, grpc_port=6334)
    collection_name = "spladev3_thresh_0_3"
    res = client.get_collection(collection_name)
    print(f"Number of vectors: {res.points_count}")
    print(f"Collection status: {res.status}")
