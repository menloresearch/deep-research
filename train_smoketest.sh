PROMPT_KEY=question
TRAIN_BATCH_SIZE=1
PPO_MINI_BATCH_SIZE=1
LR=1e-5
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=512
USE_RE_CALL=True
PROMPT_TEMPLATE_NAME=re_call_template_sys
ACTOR_MODEL_PATH="Qwen/Qwen3-0.6B"
ROLLOUT_NAME=vllm_qwen3-0.6b_with_tool_smoketest
REWARD_MANAGER=re_call
ROLLOUT_N=1
ROLLOUT_TP=1
ROLLOUT_GPU_UTIL=0.2
MAX_TURNS=5
SEARCH_URL=http://mock_retriever:8003/search
SANDBOX_URL=http://sandbox_env:8005/execute
PROJECT_NAME=deep-research
EXPERIMENT_NAME=train-qwen3-0.6b-vllm-with-tool-smoketest
NNODES=1
N_GPUS_PER_NODE=1
CUDA_VISIBLE_DEVICES_LIST="0"
SAVE_FREQ=5
TEST_FREQ=5
TOTAL_EPOCHS=1
WANDB_API_KEY="None"
SAVE_PATH="./checkpoints/train-qwen3-0.6b-vllm-with-tool-smoketest"
TRAIN_FILES="['data/processed/train.parquet', 'data/ReCall-data/syntool_re_call_sampled/train.parquet']"
TEST_FILES="['data/processed/test.parquet', 'data/ReCall-data/syntool_re_call_sampled/test.parquet']"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train_files) TRAIN_FILES="$2"; shift 2;;
        --test_files) TEST_FILES="$2"; shift 2;;
        --prompt_key) PROMPT_KEY="$2"; shift 2;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2;;
        --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2;;
        --lr) LR="$2"; shift 2;;
        --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2;;
        --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2;;
        --use_re_call) USE_RE_CALL="$2"; shift 2;;
        --prompt_template_name) PROMPT_TEMPLATE_NAME="$2"; shift 2;;
        --actor_model_path) ACTOR_MODEL_PATH="$2"; shift 2;;
        --rollout_name) ROLLOUT_NAME="$2"; shift 2;;
        --max_turns) MAX_TURNS="$2"; shift 2;;
        --reward_manager) REWARD_MANAGER="$2"; shift 2;;
        --rollout_n) ROLLOUT_N="$2"; shift 2;;
        --rollout_tp) ROLLOUT_TP="$2"; shift 2;;
        --rollout_gpu_util) ROLLOUT_GPU_UTIL="$2"; shift 2;;
        --search_url) SEARCH_URL="$2"; shift 2;;
        --sandbox_url) SANDBOX_URL="$2"; shift 2;;
        --project_name) PROJECT_NAME="$2"; shift 2;;
        --experiment_name) EXPERIMENT_NAME="$2"; shift 2;;
        --nnodes) NNODES="$2"; shift 2;;
        --n_gpus_per_node) N_GPUS_PER_NODE="$2"; shift 2;;
        --cuda_visible_devices) CUDA_VISIBLE_DEVICES_LIST="$2"; shift 2;;
        --save_freq) SAVE_FREQ="$2"; shift 2;;
        --test_freq) TEST_FREQ="$2"; shift 2;;
        --total_epochs) TOTAL_EPOCHS="$2"; shift 2;;
        --wandb_api_key) WANDB_API_KEY="$2"; shift 2;;
        --save_path) SAVE_PATH="$2"; shift 2;;
        *)
            echo "unknown argument '$1'" >&2
            exit 1;;
    esac
done

if [ "$WANDB_API_KEY" != "None" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p $SAVE_PATH
fi

ROLLOUT_SAVE_PATH=${SAVE_PATH}/rollout
if [ ! -d "$ROLLOUT_SAVE_PATH" ]; then
    mkdir -p $ROLLOUT_SAVE_PATH
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_LIST}"

SCRIPT_DIR_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT_DIR="${SCRIPT_DIR_PATH}" # Adjusted for top-level script
THIRD_PARTY_DIR="${PROJECT_ROOT_DIR}/third_party"
export PYTHONPATH="${PROJECT_ROOT_DIR}:${THIRD_PARTY_DIR}:${PYTHONPATH}"

export HYDRA_FULL_ERROR=1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$TEST_FILES" \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.use_re_call=${USE_RE_CALL} \
    data.prompt_template_name=${PROMPT_TEMPLATE_NAME} \
    data.search_url=${SEARCH_URL} \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.max_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.sandbox_url=${SANDBOX_URL} \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=${REWARD_MANAGER} \
    trainer.critic_warmup=0 \
    trainer.logger="[console, tensorboard]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.rollout_save_path=${ROLLOUT_SAVE_PATH} \
    hydra.run.dir=${SAVE_PATH}/outputs | tee ${SAVE_PATH}/run.log 