# Pre-trained Model Weights

**Status: no pre-trained weights are currently published in this repository or in a GitHub release.**
Scripts that reference `models/pretrained/quiroga_full_supervision.h5` expect a file that is not available; train your own weights with `train.py` instead.

## Planned artifact: CNN-BiLSTM, Quiroga, full supervision

- **File:** `quiroga_full_supervision.h5` (not yet released)
- **Dataset:** Quiroga et al. 2004
- **Training date:** 2024-11-15
- **Random seed:** 42
- **Software:** TensorFlow 2.8.0, CUDA 11.2.2, cuDNN 8.1.1 (see `requirements.txt`)

Performance figures will be added to this card only after the weights are published and the numbers have been re-verified against the released file.

## Reproducing results without pre-trained weights

```bash
python train.py --dataset quiroga --num-classes 6 --model-dir models/quiroga --data-dir data/quiroga
python evaluate.py --model-weights models/quiroga/best_model_quiroga.h5 \
  --test-data data/quiroga/quiroga_preprocessed.npz --num-classes 6
```

## Known limitations

- Validated only on rodent cortical data
- Performance degrades on hippocampal/primate data without fine-tuning
- Requires a GPU for practical latency
