#### roberta-base vs roberta-large on H100 80GB NVIDIA GPU (mteb/banking77)

| Model         | Batch Size (Max) | Allocated GPU Memory | Reserved GPU Memory | Inference Speed (Batch Size 1) | Inference Speed (Max Batch Size) |
| ------------- | ---------------- | -------------------- | ------------------- | ------------------------------ | -------------------------------- |
| Roberta-large | 165              | 36.35 GB             | 77.39 GB            | 98.1281 ms                     | 9.7449 ms                        |
| Roberta-base  | 433              | 35.09 GB             | 75.04 GB            | 16.3574 ms                     | 5.1870 ms                        |

#### roberta-base vs roberta-large on L40S 48GB NVIDIA GPU (mteb/banking77)

| Model         | Batch Size (Max) | Allocated GPU Memory | Reserved GPU Memory | Inference Speed (Batch Size 1) | Inference Speed (Max Batch Size) |
| ------------- | ---------------- | -------------------- | ------------------- | ------------------------------ | -------------------------------- |
| Roberta-large | 93              | 21.07 GB             | 43.88 GB            | 331.7263 ms                     | 10.4585 ms                        |
| Roberta-base  | 241              | 19.73 GB             | 42.11 GB            | 173.5017 ms                     | 5.6834 ms                        |
