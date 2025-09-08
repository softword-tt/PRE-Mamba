# !/bin/sh
export CUDA_VISIBLE_DEVICES=2,

echo "====开始运行===="
echo "[INFO] || 当前可见显卡：$CUDA_VISIBLE_DEVICES"
EXP_NAME=('SYTHETIC')
CONFIG=('PRE_Mamba')

# 检查数组长度是否一致
if [ ${#EXP_NAME[@]} -ne ${#CONFIG[@]} ]; then
    echo "Error: EXP_NAME and CONFIG arrays have different lengths"
    exit 1
fi

# 遍历所有实验配置
for ((i=0; i<${#EXP_NAME[@]}; i++)); do
    exp="${EXP_NAME[$i]}"
    config="${CONFIG[$i]}"
    
    echo "========================================"
    echo "Running test with: -n $exp -c $config"
    echo "========================================"
    
    bash scripts/test.sh \
        -g 1 \
        -n "$exp" \
        -w model_best \
        -c "$config"
done