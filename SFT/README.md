# 0 SFT(有监督微调)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在进行大模型的有监督微调（SFT）时，多轮对话的输入（input）和标签（label）需要精心组织，以确保模型能理解上下文并生成正确的回复。以下是具体的组织和案例分析：<br>

# 1 核心思路
- 输入（Input）：包含完整的对话历史，即用户与助手交替的对话内容，使用特殊标记（如<user>和<assistant>）区分角色。

- 标签（Label）：仅包含助手当前的回复内容，输入部分的所有其他内容在计算损失时被忽略（通过设置-100或类似掩码）。

- 样本拆分：每个助手的回复作为一个独立训练样本，输入为当前轮次之前的全部对话历史。

# 2 具体案例分析

假设有如下多轮对话：

```python
dialogue = [
    {"role": "user", "content": "帮我写一首诗。"},
    {"role": "assistant", "content": "春风拂面花开放，碧水潺潺鸟语香。愿君此间常驻足，心随美景共徜徉。"},
    {"role": "user", "content": "再写一首关于夏天的。"},
    {"role": "assistant", "content": "夏日炎炎荷满塘，蝉鸣树梢午梦长。轻摇蒲扇寻凉意，一片冰心在玉壶。"}
]
```

## 2.1 样本1：

- input: 用户的第一条消息 + <assistant> token

```python
<user> 帮我写一首诗。 <assistant>
```

- label: 助手的第一条消息

```python
春风拂面花开放，碧水潺潺鸟语香。愿君此间常驻足，心随美景共徜徉。
```

## 2.2 样本2：
- input: 完整的第一轮对话 + 用户的第二条消息 + 助手开始标记

```python
<user> 帮我写一首诗。 <assistant> 春风拂面花开放，碧水潺潺鸟语香。愿君此间常驻足，心随美景共徜徉。 <user> 再写一首关于夏天的。 <assistant>
```

- label: 助手的第二条消息

```python
夏日炎炎荷满塘，蝉鸣树梢午梦长。轻摇蒲扇寻凉意，一片冰心在玉壶。
```

# 3 输入与标签的token 化

以样本1为例，使用分词器（Tokenizer）处理：

**ste11:** 拼接输入与回复

完整序列：

```python
<user> 帮我写一首诗。 <assistant> 春风拂面花开放，碧水潺潺鸟语香。愿君此间常驻足，心随美景共徜徉。
```

**ste12:** 生成input_ids与labels <br>

- input_ids：整个序列的Token ID。<br>
- labels：将输入部分（<user>...<assistant>）设为-100，仅保留回复部分。<br>

```python
输入的Token： [user_token, tokens("帮我写一首诗。"), assistant_token, tokens("春风...")]
标签的Token： [-100, ..., -100, tokens("春风...")]
```

# 4 代码示例
```python
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

tokenizer = AutoTokenizer.from_pretrained("bloom-560m")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 构造样本1的输入和标签
input_text = "<user> 帮我写一首诗。 <assistant>"
response_text = "春风拂面花开放，碧水潺潺鸟语香。愿君此间常驻足，心随美景共徜徉。"

# 拼接输入和回复
full_text = input_text + response_text
tokenized = tokenizer(full_text, return_tensors="pt")

# 生成labels：输入部分设为-100，回复部分保留
input_len = len(tokenizer(input_text).input_ids)
labels = [-100] * input_len + tokenizer(response_text).input_ids

# 最终训练数据
train_sample = {
    "input_ids": tokenized["input_ids"],
    "attention_mask": tokenized["attention_mask"],
    "labels": labels
}
```

# 5 如何进行标签掩码

## 5.1 `-100` 的核心作用

### 损失掩码（Loss Masking）机制
在监督微调（SFT）中，`-100` 是损失计算的**忽略标识符**，作用如下：

1. **训练时忽略输入部分**
   - 输入内容（对话历史）的 Token 在 `labels` 中被标记为 `-100`
   - 模型仅优化非 `-100` 的 Token（即助手的回复部分）

2. **数学原理**
   交叉熵损失函数默认忽略 `-100`，公式如下：
   $$
   \text{Loss} = -\sum_{i \notin \text{ignore\_index}} y_i \log(p_i)
   $$
   其中 `ignore_index` 的默认值为 `-100`。

3. **防止干扰训练**
   若不对输入掩码，模型会尝试复现输入内容（如用户消息），而非生成目标回复。


## 5.2. `-100` 的替换可能性

### 2.1 理论允许替换
| 条件                 | 说明                                                                 |
|----------------------|--------------------------------------------------------------------|
| **框架支持**          | 需确保训练代码支持自定义 `ignore_index`（如 PyTorch 的 `CrossEntropyLoss`） |
| **值域冲突规避**      | 掩码值需不在正常 Token ID 范围内（如词汇表大小为 5 万时，掩码值应 `<0` 或 `>=5万`） |


## 5.3 代码示例

```python
import torch
import torch.nn as nn
# 自定义损失函数（ignore_index=0）
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

# 修改 DataCollator 的掩码值
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    ignore_index=-100  # 关键修改点
)

# 生成 labels 时用 0 替代 -100
input_len = len(tokenizer(input_text).input_ids)
labels = [-100] * input_len + tokenizer(response_text).input_ids
```

