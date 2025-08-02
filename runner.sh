#!/bin/bash

# Default values
model=""
quant=""
base_url="http://127.0.0.1:21434/v1"  # Default OLLaMA endpoint

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Run LocalAIME benchmark for GGUF quantized models"
    echo ""
    echo "Options:"
    echo "  --model <name>     Model name (e.g. 'Mistral-7B')"
    echo "  --quant <name>     Quantization author (e.g. 'unsloth')"
    echo "  --base-url <url>   Ollama base URL (default: $base_url)"
    echo "  --help             Show this help message"
    echo ""
    echo "Quantization Variants Tested: Q4_K_M, Q5_K_M, Q6_K, Q8_0 (unsloth specific quants: Q4_K_XL, Q5_K_XL, Q6_K_XL, Q8_K_XL)"
    echo ""
    echo "Example:"
    echo "  $0 --model TheBloke/Mistral-7B-v0.2 --quant q4_k"
    exit 0
}

# Parse arguments simply
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) model="$2"; shift 2 ;;
        --model=*) model="${1#*=}"; shift ;;
        --quant) quant="$2"; shift 2 ;;
        --quant=*) quant="${1#*=}"; shift ;;
        --base-url) base_url="$2"; shift 2 ;;
        --base-url=*) base_url="${1#*=}"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate required parameters
if [[ -z "$model" ]]; then
    echo "ERROR: --model argument is required"
    exit 1
fi

if [[ -z "$quant" ]]; then
    echo "ERROR: --quant argument is required"
    exit 1
fi

echo "Running benchmark:"
echo "• Model: $model"
echo "• Quantization: $quant"
echo "• Endpoint: $base_url"

# Define quantization variants
base_variants=(
    "hf.co/quant_name/model_name-GGUF:Q4_K_M"
    "hf.co/quant_name/model_name-GGUF:Q5_K_M"
    "hf.co/quant_name/model_name-GGUF:Q6_K"
    "hf.co/quant_name/model_name-GGUF:Q8_0"
)
unsloth_variants=(
    "hf.co/quant_name/model_name-GGUF:Q4_K_XL"
    "hf.co/quant_name/model_name-GGUF:Q5_K_XL"
    "hf.co/quant_name/model_name-GGUF:Q6_K_XL"
    "hf.co/quant_name/model_name-GGUF:Q8_K_XL"
)

# Select variants based on quant group
if [[ "$quant" == "unsloth" ]]; then
    variants=("${base_variants[@]}" "${unsloth_variants[@]}")
else
    variants=("${base_variants[@]}")
fi

# Process each variant
for variant in "${variants[@]}"; do
    model_path="hf.co/$quant/$model-GGUF:$variant"
    
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " Processing: $variant"
    echo " Model path: $model_path"
    
    # Pull model
    ollama pull "$model_path" || continue
    
    # Run benchmark
    python3 src/main.py \
        --base-url "$base_url" \
        --model "$model_path" \
        --max-tokens 32000 \
        --timeout 2000 \
        --problem-tries 3
done

echo
echo "✅ All variants processed!"

