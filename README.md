# ChatGPT-related Papers
This is a list of ChatGPT-related papers. Any feedback is welcome.

## Table of Contents
- [Survey paper](#survey-paper)
- [Instruction tuning](#instruction-tuning)
- [Reinforcement learning from human feedback](#reinforcement-learning-from-human-feedback)
- [Reinforcement learning with verifiable rewards](#reinforcement-learning-with-verifiable-rewards)
- [Reinforcement learning without verifiable rewards](#reinforcement-learning-without-verifiable-rewards) 
- [Evaluation](#evaluation)
- [Large Language Model](#large-language-model)
- [External tools](#external-tools)
- [Agent](#agent)
- [MoE/Routing](#moerouting)
- [Technical report of open/proprietary model](#technical-report-of-openproprietary-model)
- [Misc.](#misc)

## Survey paper
- [Challenges and Applications of Large Language Models](https://arxiv.org/abs/2307.10169)
- [Large Language Models: A Survey](https://arxiv.org/abs/2402.06196)

## Instruction tuning
- [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)
- [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)
- [Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks](https://arxiv.org/abs/2204.07705)
- [Self-Instruct: Aligning Language Model with Self Generated Instructions](https://arxiv.org/abs/2212.10560) [[github](https://github.com/yizhongw/self-instruct)]
- Stanford Alpaca: An Instruction-following LLaMA Model [[github](https://github.com/tatsu-lab/stanford_alpaca)]
- Dolly: Democratizing the magic of ChatGPT with open models [[blog](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html)] [[blog](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)]
- Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality [[github](https://github.com/lm-sys/FastChat)] [[website](https://vicuna.lmsys.org/)]
- [LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instructions](https://arxiv.org/abs/2304.14402) [[github](https://github.com/mbzuai-nlp/LaMini-LM)]
- [Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision](https://arxiv.org/abs/2305.03047)
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- [Enhancing Chat Language Models by Scaling High-quality Instructional Conversations](https://arxiv.org/abs/2305.14233) [[github](https://github.com/thunlp/UltraChat)]
- [How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources](https://arxiv.org/abs/2306.04751) [[github](https://github.com/allenai/open-instruct)]
- [Faith and Fate: Limits of Transformers on Compositionality](https://arxiv.org/abs/2305.18654)
- [SAIL: Search-Augmented Instruction Learning](https://arxiv.org/abs/2305.15225)
- [The False Promise of Imitating Proprietary LLMs](https://arxiv.org/abs/2305.15717)
- [Instruction Mining: High-Quality Instruction Data Selection for Large Language Models](https://arxiv.org/abs/2307.06290)
- [SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF](https://arxiv.org/abs/2310.05344) (EMNLP2023 Findings)
 
## Reinforcement learning from human feedback
- [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) [[github](https://github.com/openai/lm-human-preferences)] [[blog](https://openai.com/blog/fine-tuning-gpt-2/)]
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) [[github](https://github.com/openai/following-instructions-human-feedback)] [[blog](https://openai.com/blog/instruction-following/)]
- [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332) [[blog](https://openai.com/blog/webgpt/)]
- [Improving alignment of dialogue agents via targeted human judgements](https://arxiv.org/abs/2209.14375)
- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)
- [OpenAssistant Conversations -- Democratizing Large Language Model Alignment](https://arxiv.org/abs/2304.07327) [[github](https://github.com/LAION-AI/Open-Assistant)]
- [Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2307.15217)
- [Preference Ranking Optimization for Human Alignment](https://arxiv.org/abs/2306.17492)
- [Training Language Models with Language Feedback](https://arxiv.org/abs/2204.14146) (ACL2022 WS)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study](https://arxiv.org/abs/2404.10719)
- [HybridFlow: A Flexible and Efficient RLHF Framewor](https://arxiv.org/abs/2409.19256)
- [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036)

## Reinforcement learning with verifiable rewards
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://www.arxiv.org/abs/2505.03335)
- [Putting the Value Back in RL: Better Test-Time Scaling by Unifying LLM Reasoners With Verifiers](https://arxiv.org/abs/2505.04842)
- [Reinforcement Learning for Reasoning in Large Language Models with One Training Example](https://arxiv.org/abs/2504.20571)
- [REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2505.24760)
- [Magistral](https://mistral.ai/static/research/magistral.pdf)

## Reinforcement learning without verifiable rewards
- [Reinforcing General Reasoning without Verifiers](https://arxiv.org/abs/2505.21493)
- [Learning to Reason without External Rewards](https://arxiv.org/abs/2505.19590)
- [Can Large Reasoning Models Self-Train?](https://arxiv.org/abs/2505.21444)
 
## Evaluation
- [How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection](https://arxiv.org/abs/2301.07597)
- [Is ChatGPT a General-Purpose Natural Language Processing Task Solver?](https://arxiv.org/abs/2302.06476)
- [A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity](https://arxiv.org/abs/2302.04023)
- [Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent](https://arxiv.org/abs/2304.09542)
- [Is ChatGPT a Good Causal Reasoner? A Comprehensive Evaluation](https://arxiv.org/abs/2305.07375)
- [Is ChatGPT a Good Recommender? A Preliminary Study](https://arxiv.org/abs/2304.10149)
- [Evaluating ChatGPT's Information Extraction Capabilities: An Assessment of Performance, Explainability, Calibration, and Faithfulness](https://arxiv.org/abs/2304.11633)
- [Semantic Compression With Large Language Models](https://arxiv.org/abs/2304.12512)
- [Human-like Summarization Evaluation with ChatGPT](https://arxiv.org/abs/2304.02554)
- [Sentence Simplification via Large Language Models](https://arxiv.org/abs/2302.11957)
- [Capabilities of GPT-4 on Medical Challenge Problems](https://arxiv.org/abs/2303.13375)
- [Do Multilingual Language Models Think Better in English?](https://arxiv.org/abs/2308.01223)
- [ChatGPT or Grammarly? Evaluating ChatGPT on Grammatical Error Correction Benchmark](https://arxiv.org/abs/2303.13648)
- [ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks](https://arxiv.org/abs/2303.15056)
- [Open-Source Large Language Models Outperform Crowd Workers and Approach ChatGPT in Text-Annotation Tasks](https://arxiv.org/abs/2307.02179)
- [Can ChatGPT Reproduce Human-Generated Labels? A Study of Social Computing Tasks](https://arxiv.org/abs/2304.10145)
- [Artificial Artificial Artificial Intelligence: Crowd Workers Widely Use Large Language Models for Text Production Tasks](https://arxiv.org/abs/2306.07899)
- [Is GPT-3 a Good Data Annotator?](https://arxiv.org/abs/2212.10450) (ACL2023)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
### Benchmarks
#### General
- [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300) (ICLR2021)
- [MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark](https://arxiv.org/abs/2406.01574)
- [Are We Done with MMLU?](https://arxiv.org/abs/2406.04127) (NAACL2025)
- [Measuring short-form factuality in large language models](https://arxiv.org/abs/2411.04368)
#### Coding
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770) (ICLR2024)
- [Multi-SWE-bench: A Multilingual Benchmark for Issue Resolving](https://arxiv.org/abs/2504.02605)
- [LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code](https://arxiv.org/abs/2403.07974)
- [OJBench: A Competition Level Code Benchmark For Large Language Models](https://arxiv.org/abs/2506.16395)
 
## Large Language Model
- See https://github.com/tomohideshibata/BERT-related-papers#large-language-model

## External tools
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
- [Large Language Models as Tool Makers](https://arxiv.org/abs/2305.17126)
- [CREATOR: Disentangling Abstract and Concrete Reasonings of Large Language Models through Tool Creation](https://arxiv.org/abs/2305.14318)
- [ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](https://arxiv.org/abs/2307.16789)

## Agent
- [A Survey on Large Language Model based Autonomous Agents](https://arxiv.org/abs/2308.11432)
- [The Rise and Potential of Large Language Model Based Agents: A Survey](https://arxiv.org/abs/2309.07864)
- [Large Language Model Agent: A Survey on Methodology, Applications and Challenges](https://arxiv.org/abs/2503.21460)
- [Survey on Evaluation of LLM-based Agents](https://arxiv.org/abs/2503.16416)
 
## MoE/Routing
- [Routing to the Expert: Efficient Reward-guided Ensemble of Large Language Models](https://arxiv.org/abs/2311.08692)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [Knowledge Fusion of Large Language Models](https://arxiv.org/abs/2401.10491) (ICLR2024)

## Technical report of open/proprietary model
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [Qwen Technical Report](https://arxiv.org/abs/2309.16609)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [Nemotron-4 15B Technical Report](https://arxiv.org/abs/2402.16819)
- [Nemotron-4 340B Technical Report](https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_4_340B_8T_0.pdf)
- [PaLM 2 Technical Report](https://arxiv.org/abs/2305.10403)
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)
- [Hunyuan-A13B Technical Report](https://github.com/Tencent-Hunyuan/Hunyuan-A13B/blob/main/report/Hunyuan_A13B_Technical_Report.pdf)
- [ERNIE 4.5 Technical Report](https://yiyan.baidu.com/blog/publication/ERNIE_Technical_Report.pdf)
 
## Misc.
- [Summary of ChatGPT/GPT-4 Research and Perspective Towards the Future of Large Language Models](https://arxiv.org/abs/2304.01852)
- [Don't Trust GPT When Your Question Is Not In English](https://arxiv.org/abs/2305.16339)
- [GPT4GEO: How a Language Model Sees the World's Geography](https://arxiv.org/abs/2306.00020)
- [ChatGPT: A Meta-Analysis after 2.5 Months](https://arxiv.org/abs/2302.13795)
