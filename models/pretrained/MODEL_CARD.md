```markdown
# Pre-trained Model Weights

## CNN-BiLSTM Quiroga Full Supervision
- **File:** `quiroga_full_supervision.h5`
- **Size:** ~3.8 MB
- **Training Date:** 2024-03-15
- **Hardware:** NVIDIA RTX 3090, Driver 510.47.03
- **Software:** TensorFlow 2.8.0, CUDA 11.2.2, cuDNN 8.1.1
- **Dataset:** Quiroga et al. 2004 (4 neurons, ~10,000 spikes)
- **Random Seed:** 42

### Verified Performance
Run `python evaluate.py --model-weights models/pretrained/quiroga_full_supervision.h5` to verify:
- Accuracy: 0.964 [0.962, 0.967]
- Precision: 0.948 [0.945, 0.952]
- Latency (RTX 3090): 45.2 ms amortized / 52.1 ms single-spike

### Limitations
- Validated only on rodent cortical data
- Performance degrades on hippocampal/primate data without fine-tuning
- Requires GPU with >= 10GB VRAM
