# Module 10: Advanced AI, Graph Neural Networks & Large Language Models

[Back to the master roadmap](../README.md) · [Environment setup](../docs/setup.md)

**Weeks 51–54 · October 18–November 14, 2027 · Status: Planned**

## Overview

This final module connects reinforcement learning, graph neural networks, point clouds, multimodal large language models, retrieval, agents, and deployment. It is deliberately integrative: the capstone should combine at least two earlier modules into a system that can be evaluated, explained, and reproduced.

The goal is not to cover every advanced topic superficially. It is to build the judgment to choose the right representation, model class, retrieval strategy, and evaluation method for a new problem.

## Learning Outcomes

By the end of this module, you should be able to:

- Formulate a sequential decision problem as a Markov decision process and train a basic RL agent.
- Explain message passing, graph convolutions, and graph-level prediction tasks.
- Work with point-cloud or 3D representations at a foundational level.
- Build a multimodal LLM pipeline that combines text with another modality.
- Implement retrieval-augmented generation with citations and retrieval evaluation.
- Design a tool-using or multi-agent workflow with explicit failure handling.
- Deliver a tested, documented, deployable capstone that integrates multiple modules.

## Weekly Plan

| Week | Dates | Focus | Output |
| :---: | :--- | :--- | :--- |
| 51 | Oct 18–24, 2027 | Reinforcement learning, MDPs, Q-learning, policy gradients, and DDPG | RL and point-cloud lab |
| 52 | Oct 25–31, 2027 | Graph neural networks, molecular property prediction, and multimodal LLMs | Graph and multimodal lab |
| 53 | Nov 1–7, 2027 | RAG, tool use, agents, multi-agent systems, and deployment | Retrieval and agent lab |
| 54 | Nov 8–14, 2027 | Capstone integration, evaluation, documentation, portfolio, and retrospective | Applied AI Capstone milestone |

## Milestone Project: Applied AI Capstone

Build an end-to-end AI system that integrates at least two modules. Examples include a document assistant with retrieval and evaluation, a graph-based molecular property predictor with an interactive explorer, or a multimodal search system with a deployed API.

### Requirements

- A precise problem statement, intended user, and success criteria.
- A dataset or knowledge source with license, provenance, and privacy review.
- A baseline and at least one improved system.
- An evaluation protocol with task metrics and failure analysis.
- At least two integrated components, such as retrieval plus an LLM, a GNN plus a prediction service, or vision plus language.
- Tests for data processing, core logic, and API behavior.
- A reproducible environment and a documented run command.
- A deployed local demo using FastAPI, Gradio, or Streamlit.
- A model or system card covering limitations, bias, security, and misuse.
- A portfolio-ready README and a short technical presentation.

### Stretch Goals

- Add multi-agent orchestration with explicit tool permissions.
- Add a graph neural network for a real relational dataset.
- Add a video or 3D point-cloud component.
- Compare RAG with and without reranking.
- Evaluate robustness to prompt injection, missing evidence, or noisy retrieval.
- Measure latency, cost, and failure rates under realistic input variation.

## Workspace

```text
10-advanced-deep-learning-and-llms/
├── notebooks/
│   ├── 51-reinforcement-learning.ipynb
│   ├── 52-graphs-and-multimodal-llms.ipynb
│   └── 53-rag-and-agents.ipynb
├── exercises/
└── projects/
    └── applied-ai-capstone/
```

## Tools

- PyTorch 2.x
- PyTorch Geometric
- NetworkX
- Gymnasium
- Hugging Face Transformers
- `sentence-transformers`
- FAISS
- LangChain or LangGraph
- FastAPI, Gradio, or Streamlit
- MLflow
- ONNX Runtime (optional)

## Deep Dives

- Deep deterministic policy gradients.
- Point-cloud learning and 3D representations.
- Multi-task learning.
- Retrieval evaluation and reranking.
- Agent evaluation, tool safety, and prompt-injection defenses.

## Assessment

| Criterion | Weight | What good looks like |
| :--- | :---: | :--- |
| Problem and scope | 15% | The system solves a real, well-scoped problem with clear success criteria. |
| Integration | 20% | Multiple components work together coherently rather than as a demo collage. |
| Modeling and retrieval | 20% | Model and retrieval choices are justified and compared to a baseline. |
| Evaluation and testing | 20% | Metrics, error analysis, robustness checks, and automated tests are present. |
| Reproducibility and deployment | 15% | The system runs from a documented command and has a working demo. |
| Communication and ethics | 10% | Documentation covers limitations, bias, security, and responsible use. |

## References

- *Reinforcement Learning: An Introduction* by Richard S. Sutton and Andrew G. Barto.
- *Graph Representation Learning* by William L. Hamilton.
- *Semi-Supervised Classification with Graph Convolutional Networks* by Thomas N. Kipf and Max Welling.
- *Attention Is All You Need* by Ashish Vaswani et al.
- *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* by Patrick Lewis et al.
- *ReAct: Synergizing Reasoning and Acting in Language Models* by Shunyu Yao et al.
- PyTorch Geometric documentation: <https://pytorch-geometric.readthedocs.io/>
- LangGraph documentation: <https://langchain-ai.github.io/langgraph/>

## Module Checklist

- [ ] All weekly notebooks run from a clean kernel.
- [ ] The capstone integrates at least two modules.
- [ ] A baseline and an improved system are compared.
- [ ] The evaluation includes failure analysis and robustness checks.
- [ ] Automated tests cover data processing, core logic, and API behavior.
- [ ] The demo runs from a documented command.
- [ ] The system card addresses bias, security, privacy, and misuse.
- [ ] The final retrospective and portfolio entry are complete.
