python extract_librispeech_features.py \
    -i /mnt/wsl/nvme/datasets/LibriSpeech/ \
    -o /mnt/wsl/nvme/data/LibriSpeech/features/wavlm-large/ \
    -s dev-clean \
    -s dev-other \
    -s train-clean-100 \
    -l 24