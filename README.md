# ChatGPT-related Papers
This is a list of ChatGPT-related papers. Any feedback is welcome.

## Table of Contents
- [Survey paper](#survey-paper)
- [Pre-training](#pre-training) 
- [Instruction tuning](#instruction-tuning)
- [Reinforcement learning from human feedback](#reinforcement-learning-from-human-feedback)
- [Reinforcement learning with verifiable rewards](#reinforcement-learning-with-verifiable-rewards)
- [Reinforcement learning without verifiable rewards](#reinforcement-learning-without-verifiable-rewards) 
- [Evaluation](#evaluation)
- [Large Language Model](#large-language-model)
- [External tools](#external-tools)
- [Agent](#agent)
- [Autonomous learning](#autonomous-learning)
- [MoE/Routing](#moerouting)
- [Technical report of open/proprietary model](#technical-report-of-openproprietary-model)
- [Misc.](#misc)

## Survey paper
- [Challenges and Applications of Large Language Models](https://arxiv.org/abs/2307.10169)
- [Large Language Models: A Survey](https://arxiv.org/abs/2402.06196)

## Pre-training
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)
- [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557)
- [FineWeb2: One Pipeline to Scale Them All -- Adapting Pre-Training Data Processing to Every Language](https://arxiv.org/abs/2506.20920) (COLM2025)
- [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794)
- [OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text](https://arxiv.org/abs/2310.06786)
- [InfiMM-WebMath-40B: Advancing Multimodal Pre-Training for Enhanced Mathematical Reasoning](https://arxiv.org/abs/2409.12568)
- [The Stack: 3 TB of permissively licensed source code](https://arxiv.org/abs/2211.15533)
- [StarCoder: may the source be with you!](https://arxiv.org/abs/2305.06161)
- [StarCoder 2 and The Stack v2: The Next Generation](https://arxiv.org/abs/2402.19173)
- [To Code, or Not To Code? Exploring Impact of Code in Pre-training](https://arxiv.org/abs/2408.10914)
- [Arctic-SnowCoder: Demystifying High-Quality Data in Code Pretraining](https://arxiv.org/abs/2409.02326)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) (EMNLP2023)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies](https://arxiv.org/abs/2407.13623) (NeurIPS2024)
 
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
- [Faith and Fate: Limits of Transformers on Compositionality](https://arxiv.org/abs/2305.18654)
- [SAIL: Search-Augmented Instruction Learning](https://arxiv.org/abs/2305.15225)
- [The False Promise of Imitating Proprietary LLMs](https://arxiv.org/abs/2305.15717)
- [Instruction Mining: High-Quality Instruction Data Selection for Large Language Models](https://arxiv.org/abs/2307.06290)
- [SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF](https://arxiv.org/abs/2310.05344) (EMNLP2023 Findings)
- [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)
 
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
- [A Survey of Reinforcement Learning for Large Reasoning Models](https://arxiv.org/abs/2509.08827)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783) (COLM2025)
- [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
- [Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback](https://arxiv.org/abs/2506.03106)
- [Sample More to Think Less: Group Filtered Policy Optimization for Concise Reasoning](https://www.arxiv.org/abs/2508.09726)
- [Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains](https://arxiv.org/abs/2507.17746)
- [Checklists Are Better Than Reward Models For Aligning Language Models](https://arxiv.org/abs/2507.18624)
- [Putting the Value Back in RL: Better Test-Time Scaling by Unifying LLM Reasoners With Verifiers](https://arxiv.org/abs/2505.04842)
- [Reinforcement Learning for Reasoning in Large Language Models with One Training Example](https://arxiv.org/abs/2504.20571)
- [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837) (NeurIPS2025)
- [ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models](https://arxiv.org/abs/2505.24864)
- [REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2505.24760)
- [Magistral](https://mistral.ai/static/research/magistral.pdf)
- [Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning](https://arxiv.org/abs/2508.08221)
- [DeepSearch: Overcome the Bottleneck of Reinforcement Learning with Verifiable Rewards via Monte Carlo Tree Search](https://www.arxiv.org/abs/2509.25454)
- [Reasoning with Sampling: Your Base Model is Smarter Than You Think]()
 
## Reinforcement learning without verifiable rewards
- [Reinforcing General Reasoning without Verifiers](https://arxiv.org/abs/2505.21493)
- [Learning to Reason without External Rewards](https://arxiv.org/abs/2505.19590)
- [Can Large Reasoning Models Self-Train?](https://arxiv.org/abs/2505.21444)
 
## Evaluation
- [A Survey on Evaluation of Large Language Models](https://arxiv.org/abs/2307.03109)
- [Evaluating Large Language Models: A Comprehensive Survey](https://arxiv.org/abs/2310.19736)
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
- [Fluid Language Model Benchmarking](https://www.arxiv.org/abs/2509.11106) (COLM2025)
### Benchmarks
- [A Survey on Large Language Model Benchmarks](https://arxiv.org/abs/2508.15361)
#### General
- [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300) (ICLR2021)
- [MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark](https://arxiv.org/abs/2406.01574)
- [Are We Done with MMLU?](https://arxiv.org/abs/2406.04127) (NAACL2025)
- [Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation](https://arxiv.org/abs/2412.03304) (ACL2025)
- [VNHSGE: VietNamese High School Graduation Examination Dataset for Large Language Models](https://arxiv.org/abs/2305.12199)
- [CMMLU: Measuring massive multitask language understanding in Chinese](https://arxiv.org/abs/2306.09212)
- [HAE-RAE Bench: Evaluation of Korean Knowledge in Language Models](https://arxiv.org/abs/2309.02706)
- [KMMLU: Measuring Massive Multitask Language Understanding in Korean](https://arxiv.org/abs/2402.11548) (NAACL2025)
- [From KMMLU-Redux to KMMLU-Pro: A Professional Korean Benchmark Suite for LLM Evaluation](https://arxiv.org/abs/2507.08924)  
- [Large Language Models Only Pass Primary School Exams in Indonesia: A Comprehensive Test on IndoMMLU](https://arxiv.org/abs/2310.04928) (EMNLP2023)
- [Typhoon: Thai Large Language Models](https://arxiv.org/abs/2312.13951)
- [ArabicMMLU: Assessing Massive Multitask Language Understanding in Arabic](https://arxiv.org/abs/2402.12840) (ACL2024 Findings)
- [Khayyam Challenge (PersianMMLU): Is Your LLM Truly Wise to The Persian Language?](https://arxiv.org/abs/2404.06644)
- [IrokoBench: A New Benchmark for African Languages in the Age of Large Language Models](https://arxiv.org/abs/2406.03368)
- [TurkishMMLU: Measuring Massive Multitask Language Understanding in Turkish](https://arxiv.org/abs/2407.12402) (EMNLP2024 Findings)
- [Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models](https://arxiv.org/abs/2206.04615)
- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)
- [AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models](https://arxiv.org/abs/2304.06364)
- [C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models](https://arxiv.org/abs/2305.08322) (NeurIPS2023)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [MT-Bench-101: A Fine-Grained Benchmark for Evaluating Large Language Models in Multi-Turn Dialogues](https://arxiv.org/abs/2402.14762)
- [The BiGGen Bench: A Principled Benchmark for Fine-grained Evaluation of Language Models with Language Models](https://arxiv.org/abs/2406.05761) (NAACL2025)
- [GPQA: A Graduate-Level Google-Proof Q&A Benchmark](https://arxiv.org/abs/2311.12022)
- [SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines](https://arxiv.org/abs/2502.14739)
- [LiveBench: A Challenging, Contamination-Limited LLM Benchmark](https://arxiv.org/abs/2406.19314)
- [Measuring short-form factuality in large language models](https://arxiv.org/abs/2411.04368)
- [Humanity's Last Exam](https://arxiv.org/abs/2501.14249)
- [MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs](https://arxiv.org/abs/2501.17399)
#### Coding
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)
- [HumanEval-XL: A Multilingual Code Generation Benchmark for Cross-lingual Natural Language Generalization](https://arxiv.org/abs/2402.16694) (LREC-COLING2024)
- [Program Synthesis with Large Language Models](https://arxiv.org/abs/2108.07732)
- [MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation](https://arxiv.org/abs/2208.08227)
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770) (ICLR2024)
- [Multi-SWE-bench: A Multilingual Benchmark for Issue Resolving](https://arxiv.org/abs/2504.02605)
- [LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code](https://arxiv.org/abs/2403.07974)
- [OJBench: A Competition Level Code Benchmark For Large Language Models](https://arxiv.org/abs/2506.16395)
#### Math
- [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874)
- [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
- [Omni-MATH: A Universal Olympiad Level Mathematic Benchmark For Large Language Models](https://arxiv.org/abs/2410.07985)
- [MathArena: Evaluating LLMs on Uncontaminated Math Competitions](https://arxiv.org/abs/2505.23281)
#### Instruction following
- [Instruction-Following Evaluation for Large Language Models](https://arxiv.org/abs/2311.07911)
- [Can Large Language Models Understand Real-World Complex Instructions?](https://arxiv.org/abs/2309.09150)
- [FollowBench: A Multi-level Fine-grained Constraints Following Benchmark for Large Language Models](https://arxiv.org/abs/2310.20410) (ACL2024)
- [Generalizing Verifiable Instruction Following](https://arxiv.org/abs/2507.02833)
- [CIF-Bench: A Chinese Instruction-Following Benchmark for Evaluating the Generalizability of Large Language Models](https://arxiv.org/abs/2402.13109) (ACL2024)
- [Multi-IF: Benchmarking LLMs on Multi-Turn and Multilingual Instructions Following](https://arxiv.org/abs/2410.15553)
- [MaXIFE: Multilingual and Cross-lingual Instruction Following Evaluation](https://arxiv.org/abs/2506.01776) (ACL2025)
#### Agent
- [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045)
#### Long-context
- [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654) (COLM2024)
- [LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks](https://arxiv.org/abs/2412.15204) (ACL2025)
- [One ruler to measure them all: Benchmarking multilingual long-context language models](https://arxiv.org/abs/2503.01996)
 
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
- [A Survey of Self-Evolving Agents: On Path to Artificial Super Intelligence](https://arxiv.org/abs/2507.21046)
- [A Comprehensive Survey of Self-Evolving AI Agents: A New Paradigm Bridging Foundation Models and Lifelong Agentic Systems](https://arxiv.org/abs/2508.07407)
- [Survey on Evaluation of LLM-based Agents](https://arxiv.org/abs/2503.16416)
- [RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents](https://arxiv.org/abs/2507.22844)

## Autonomous learning
- [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://www.arxiv.org/abs/2505.03335)
- [R-Zero: Self-Evolving Reasoning LLM from Zero Data](https://arxiv.org/abs/2508.05004)
- [Self-Adapting Language Models](https://arxiv.org/abs/2506.10943)
- [Towards Agentic Self-Learning LLMs in Search Environment](https://www.arxiv.org/abs/2510.14253)

## MoE/Routing
- [Routing to the Expert: Efficient Reward-guided Ensemble of Large Language Models](https://arxiv.org/abs/2311.08692)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [Knowledge Fusion of Large Language Models](https://arxiv.org/abs/2401.10491) (ICLR2024)

## Interpretability
- [Open Problems in Mechanistic Interpretability](https://arxiv.org/abs/2501.16496)
- [Do Llamas Work in English? On the Latent Language of Multilingual Transformers](https://arxiv.org/abs/2402.10588)
 
## Technical report of open/proprietary model
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [Qwen Technical Report](https://arxiv.org/abs/2309.16609)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [The Falcon Series of Open Language Models](https://arxiv.org/abs/2311.16867)
- [Nemotron-4 15B Technical Report](https://arxiv.org/abs/2402.16819)
- [Nemotron-4 340B Technical Report](https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_4_340B_8T_0.pdf)
- [PaLM 2 Technical Report](https://arxiv.org/abs/2305.10403)
- [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656)
- [How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources](https://arxiv.org/abs/2306.04751) [[github](https://github.com/allenai/open-instruct)]
- [Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2](https://arxiv.org/abs/2311.10702)
- [Tulu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124)
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)
- [Kimi K2: Open Agentic Intelligence](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf)
- [SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model](https://arxiv.org/abs/2502.02737) (COLM2025)
- [Hunyuan-A13B Technical Report](https://github.com/Tencent-Hunyuan/Hunyuan-A13B/blob/main/report/Hunyuan_A13B_Technical_Report.pdf)
- [ERNIE 4.5 Technical Report](https://yiyan.baidu.com/blog/publication/ERNIE_Technical_Report.pdf)
- [ESTIMATING WORST-CASE FRONTIER RISKS OF OPEN-WEIGHT LLMS](https://cdn.openai.com/pdf/231bf018-659a-494d-976c-2efdfc72b652/oai_gpt-oss_Model_Safety.pdf)
- [GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models](https://arxiv.org/abs/2508.06471)
## Misc.
- [Summary of ChatGPT/GPT-4 Research and Perspective Towards the Future of Large Language Models](https://arxiv.org/abs/2304.01852)
- [Don't Trust GPT When Your Question Is Not In English](https://arxiv.org/abs/2305.16339)
- [GPT4GEO: How a Language Model Sees the World's Geography](https://arxiv.org/abs/2306.00020)
- [ChatGPT: A Meta-Analysis after 2.5 Months](https://arxiv.org/abs/2302.13795)
