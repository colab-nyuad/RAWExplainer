code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
conda_env_path=/scratch/sl9030/conda-envs/Ultra
python_path=$conda_env_path/bin/python
SCRIPT_NAME=script/global_explanation.py
INFERENCE_CONFIG=config/new_explain/nell995-pgexplainer-inference-jubail-samyam.yaml
cd $code_dir

# Logging
echo "Running PGExplainer inference script: $SCRIPT_NAME"
echo "Using Python interpreter: $python_path"
echo "Code directory: $code_dir"
echo "Conda environment path: $conda_env_path"
echo "Inference config: $INFERENCE_CONFIG"
echo "Job started at: $(date)"
echo "Current working directory: $(pwd)"

# Single GPU inference (no need for DDP during inference)
$python_path $SCRIPT_NAME -c $INFERENCE_CONFIG --use_wandb no --gpus [0]

# Logging
echo "PGExplainer inference job finished at: $(date)"
