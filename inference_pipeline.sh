# Define the data path
data_path="dataset/contextvqa.json"

# Check if the file exists before running the script
if [[ -f "$data_path" ]]; then
    echo "Processing file: $data_path"

    ##############################
    # echo "Inference on LLAVA-3D 7B Model"
    # source /gpfs/home/ym621/mambaforge-pypy3/etc/profile.d/conda.sh
    # module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0  
    # conda activate llava-3d
    # python 3D-VLM/LLaVA-3D/llava/eval/run_llava_3d.py \
    # --model-path ChaimZhu/LLaVA-3D-7B -f "$data_path"

    # echo "Inference on Qwen2-VL 7B Model"
    # source /gpfs/home/ym621/mambaforge-pypy3/etc/profile.d/conda.sh
    # module load NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1 
    # conda activate qwen2
    python 2D-VLM/Qwen2-VL/evaluate.py -f "$data_path" -m "/mnt/pfs/zitao_team/big_model/raw_models/Qwen2-VL-7B-Instruct"

    # echo "Inference on Qwen2-VL 72B Model"
    # python 2D-VLM/Qwen2-VL/evaluate.py -f "$data_path" -m "Qwen/Qwen2-VL-72B-Instruct-AWQ"

    # echo "Inference on llava-ov 7B Model"
    # python 2D-VLM/llava-ov/evaluate.py -f "$data_path" -m "llava-hf/llava-onevision-qwen2-7b-ov-hf"

    # echo "Inference on llava-ov 72B Model"
    # python 2D-VLM/llava-ov/evaluate.py -f "$data_path" -m "llava-hf/llava-onevision-qwen2-72b-ov-hf"

    # echo "Inference on GPT4o-VL API Model"
    # python 2D-VLM/GPT4o/evaluate.py -f "$data_path"

    # echo "Inference on Claude-3.5-Sonnet API Model"
    # python 2D-VLM/Claude/evaluate.py -f "$data_path"

    # echo "Inference on GPT4o-text API Model"
    # python LLM/GPT4o-text/evaluate.py -f "$data_path"

    # echo Inference on llama-3.2 3B Model
    # python LLM/llama/evaluate.py -f "$data_path"
else
    echo "Error: File not found: $data_path"
fi
