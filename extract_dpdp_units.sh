python extract_dpdp_units.py \
    -i /mnt/wsl/nvme/data/LibriSpeech/features/wavlm-large/layer-24/ \
    -o /mnt/wsl/nvme/data/LibriSpeech/cluster-ids/wavlm-large/layer-24/km-500/ \
    -c ./wavlm-large-layer-24-kmeans-500-centroids.npy \
    -s dev-clean \
    -s dev-other \
    -s train-clean-100