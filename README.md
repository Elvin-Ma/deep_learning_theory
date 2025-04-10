# deep_learning_theory
> Summary of deep learning theory

## Table of Contents

- 1-feedforward_network/
- 2-back_propagation/
- 3-bp_example_demo/
- 4-convolution_neural_network/
- 5-deep_learning_model/
- 6-pytorch_install/
- 7-operators/
- 8-activation_functions/
- 9-recurrent_neural_network/
- 10-seq2seq/
- 11-attentions/
- 12-weight-initialization/
- 13-optimizers/
- 14-regularization/
- 15-deep-learning-tuning-guide/
- 16-learning_rate/
- 20-pytorch-tensor/
- 21-pytorch-autograd/
- 22-pytorch-module/
- 23-training-examples/
- 24-pytorch-optimizer/
- 25-pytorch-lr-scheduler/
- 25-pytorch-data-state/
- 26-pytorch-dataloader/
- 27-pytorch-model-save/
- 28-pytorch-tensorboard/
- 29-pytorch-graph-mode/
- 30-training_examples_cv/
- 31-stable-diffusion/
- 33-stable-diffusion/
- 40-nlp_bert/
- 41-nlp_t5/
- 42-nlp-gpt/
- 43-scaling-law/
- 44-distribute-training/
- 45-LLM-History/
- 46-LLM-GPT-Extension/
- 46-LLM-Llama/
- 47-LLM-DeepSeek-Structure/
- 48-LLM-deepseek-r1-training/
- 49-PPO-GRPO/
- 50-Chain-of-Thought/
- 100-LLM-Survey/

**other:**

- position_embedding/



---



## Lessons Additions and Adjustments

### 内容补充：

| 序号 | 补充内容                                                     | 状态 |
| :--: | :----------------------------------------------------------- | :--: |
| 001  | update：softmax 激活函数的导数引出一下（雅可比矩阵）         |  1   |
| 002  | add：Norm 讲解的时候，未加入最新的 DyT（[Transformers without normalization](https://yiyibooks.cn/arxiv/2503.10622v1/index.html)） |  1   |
| 003  | add：DeepNorm 补充                                           |  1   |
| 004  | add：PyTorch 等框架模型结构中的参数类型和数据整理~~（我的笔记）~~（已补充到课件PyTorch部分） |  0   |
| 005  | 思考：工程如何实现训练和推理不同的模块或者算子（那个 training 参数和具体的算子结构） |  0   |
| 006  |                                                              |  0   |



### 调整建议:

| 序号 | 调整建议                                                     |
| :--: | :----------------------------------------------------------- |
| 001  | updata：torch 的 Tensor 中，数据有 metadata 和 storage 之分（之前讲成 rawdata，但官网未使用这种叫法） [torch.Srorage](https://pytorch.org/docs/stable/storage.html) |
| 002  | 优化：前后知识交叉部分可以切回到原理快速回顾一下（比如：训练模式与Norm和Dropout、torch的数据结构与一些基础算子等） |
|      |                                                              |
|      |                                                              |
|      |                                                              |

