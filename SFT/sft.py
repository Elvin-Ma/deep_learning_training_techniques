# 环境准备
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
import accelerate

# 一、数据预处理模块
def load_and_format_data():
    # 加载YaoYX数据集（含61k+多轮对话样本）
    dataset = load_dataset("YaoYX/llama_instruct_sample", split="train")

    # 转换为多轮对话格式（支持多轮扩展）
    def format_conversation(example):
        conversations = []
        # 支持将"scores"字段中的多轮评分序列转换为对话轮次
        for i, (inst, out) in enumerate(zip(example["instruction"], example["output"])):
            conversations.extend([
                {"role": "user", "content": f"<round{i+1}> {inst}"},  # 添加对话轮次标记
                {"role": "assistant", "content": out}
            ])
        return {"conversations": conversations}

    return dataset.map(format_conversation,  batched=True)

# 二、模型加载模块（CPU优化版）
def load_model_cpu():
    # 内存优化配置（8-bit量化需安装bitsandbytes）
    quantization_config = None
    if False:  # 若安装bitsandbytes可将此处改为True
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # 加载模型（强制CPU模式）
    model = AutoModelForCausalLM.from_pretrained(
        "Meta-llama/Llama-3.2-1B-Instruct",
        device_map={"": "cpu"},
        torch_dtype=torch.float32,
        quantization_config=quantization_config,
    )

    # 启用梯度检查点节省内存
    model.gradient_checkpointing_enable()

    # 加载分词器（添加对话标记）
    tokenizer = AutoTokenizer.from_pretrained("Meta-llama/Llama-3.2-1B-Instruct")
    tokenizer.add_tokens(["<round1>",  "<round2>", "<round3>"])  # 扩展多轮对话标记

    return model, tokenizer

# 三、训练流程模块
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        return (outputs.loss,  outputs) if return_outputs else outputs.loss

def train_model():
    # 加载组件
    model, tokenizer = load_model_cpu()
    dataset = load_and_format_data()

    # 训练参数配置（CPU优化版）
    training_args = TrainingArguments(
        output_dir="./llama3.2_multi-turn_results",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=5e-5,
        optim="adamw_torch",
        logging_steps=50,
        save_strategy="no",
        fp16=False  # 必须关闭混合精度
    )

    # 启动训练
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda data: tokenizer.pad(
            data,
            padding="longest",
            return_tensors="pt"
        )
    )
    trainer.train()

if __name__ == "__main__":
    train_model()