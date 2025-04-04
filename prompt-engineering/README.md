# 0 概述

提示工程是一个相对较新的学科，旨在开发和优化提示(prompts)，以高效利用语言模型（LMs）应用于各种广泛的应用领域和研究主题。提示工程技能有助于更好地理解大型语言模型（LLMs）的能力和局限性。

研究人员利用提示工程来提高大型语言模型在广泛常见和复杂任务（如问题回答和算术推理）上的能力。开发人员则利用提示工程来设计稳健且有效的提示技术，以与大型语言模型和其他工具进行接口。

提示工程不仅仅涉及设计和开发提示。它涵盖了一系列与大型语言模型交互和开发相关的技能和技巧。这是与大型语言模型接口、构建和理解其能力的重要技能。你可以**利用提示工程来提高大型语言模型的安全性，并构建新的能力**，如通过领域知识和外部工具增强大型语言模型。

鉴于对利用大型语言模型进行开发的浓厚兴趣，我们编写了这份新的提示工程指南，其中包含了所有最新的论文、先进的提示技术、学习指南、模型特定的提示指南、讲座、参考文献、大型语言模型的新能力以及与提示工程相关的工具。


# 1 Introduction

提示工程是一个相对较新的学科，旨在开发和优化提示，以高效应用和构建大型语言模型（LLMs），满足各种广泛的应用场景和使用需求。

提示工程技能有助于更好地理解大型语言模型的能力和局限性。研究人员利用提示工程来提高大型语言模型在广泛常见和复杂任务（如问题回答和算术推理）上的**安全性和能力**。开发人员则利用提示工程来**设计稳健且有效的提示技术**，以与大型语言模型和其他工具进行接口。

本综合指南涵盖了提示工程的理论和实践方面，以及如何利用最佳的提示技术与大型语言模型进行交互和构建。

除非另有说明，所有示例均使用OpenAI的Playground中的gpt-3.5-turbo模型进行测试。该模型使用默认配置，即temperature=1和top_p=1。这些提示也应适用于具有与gpt-3.5-turbo相似能力的其他模型，但模型响应可能会有所不同。

# 2 LLM Settings

在设计和测试提示时，你通常通过API与大型语言模型（LLM）进行交互。你可以配置一些参数以获得不同的提示结果。调整这些设置对于提高响应的可靠性和满意度非常重要，并且需要通过一些实验来找出适合你用例的适当设置。以下是使用不同LLM提供商时你会遇到的常见设置：

温度（Temperature）- 简而言之，温度越低，结果越确定，因为总是选择下一个最可能的词汇。**提高温度可能会增加随机性**，从而鼓励更多样化或更具创意的输出。你实际上是在增加其他可能词汇的权重。在应用方面，对于基于事实的问答等任务，你可能希望使用较低的温度值，以鼓励更准确和简洁的响应。对于诗歌生成或其他创意任务，提高温度值可能更有益。

Top P - 这是一种与温度相关的采样技术，称为**核采样(nucleus sampling)**，你可以通过它来控制模型的确定性。如果你寻求准确且符合事实的答案，请将此值保持较低。如果你希望获得更多样化的响应，请将其增加到较高的值。如果你使用Top P，这意味着只有构成top_p概率质量的词汇才会被考虑用于响应，因此较低的top_p值会选择最自信的响应。这意味着较高的top_p值将使模型能够考虑更多可能的词汇，包括不太可能的词汇，从而产生更多样化的输出。

一般建议是调整温度或Top P，但不要同时调整两者。

**最大长度（Max Length）** - 你可以通过调整最大长度来管理模型生成的词汇数量。指定最大长度有助于防止生成过长或不相关的响应，并控制成本。

**停止序列（Stop Sequences）** - 停止序列是一个字符串，用于停止模型生成词汇。指定停止序列是控制模型响应长度和结构的另一种方法。例如，你可以通过添加“11”作为停止序列，告诉模型生成的列表项不超过10个。

**频率惩罚（Frequency Penalty）** - 频率惩罚对下一个词汇应用惩罚，惩罚程度与该词汇在响应和提示中已经出现的次数成正比。频率惩罚越高，词汇再次出现的可能性越小。此设置通过对出现次数较多的词汇施加更高的惩罚，减少模型响应中词汇的重复。

**存在惩罚（Presence Penalty）** - 存在惩罚也对重复词汇应用惩罚，但与频率惩罚不同的是，对所有重复词汇的惩罚是相同的。一个出现两次的词汇和一个出现10次的词汇受到的惩罚是一样的。此设置防止模型在响应中过于频繁地重复短语。如果你希望模型生成多样化或创意性的文本，可能需要使用较高的存在惩罚。或者，如果你需要模型保持专注，可以尝试使用较低的存在惩罚。

与温度和top_p类似，一般建议是调整频率惩罚或存在惩罚，但不要同时调整两者。

在开始一些基本示例之前，请记住，你的结果可能会因使用的LLM版本而异。



# 10 参考链接
- [prompt Engineering Guide](https://www.promptingguide.ai/techniques/zeroshot)