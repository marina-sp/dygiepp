# Train DyGIE++ model on the scierc data set.
# Usage: bash scripts/train/train_scierc.sh [gpu-id]
# gpu-id can be an integer GPU ID, or -1 for CPU.

experiment_name="test"

chmod u+wrx models/test/*; rm -r models/$experiment_name/

data_root="./data/kn"
config_file="./training_config/kn_local.jsonnet"
cuda_device=$1

# Train model.
ie_train_data_path=$data_root/kn.train.train.json \
    ie_dev_data_path=$data_root/kn.train.dev.json \
    ie_test_data_path=$data_root/kn.dev.json \
    cuda_device=$cuda_device \
    allennlp train $config_file \
    --cache-directory $data_root/cached \
    --serialization-dir ./models/$experiment_name \
    --include-package dygie
