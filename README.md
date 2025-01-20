# ComfyUI-EmbeddingPipelineAnalytics

## overview

this repo is to capture end-to-end data, metadata, and embeddings for ComfyUI workflows, specifically HunyuanVideo to start.

## goals

- creating more formalized datasets for open source projects and teams for analysis and/or downstream tasks, of which there are infinite experiments to explore
- giving more powerful analysis tools and data management tooling to non-developers in the community
- providing a feedback mechanism for analyzing embeddings vs. inputs

## features

as this will be maintained based on my own free time, the core will focus on the following:

*Generic capture node that can:*

- Capture embeddings and run metadata and data inputs / outputs from ComfyUI workflows (parameters, prompt, guidance, sampling, etc)
- Store data in a structured format for analysis (parquet, jsonl)

*Analysis node that can:*

- Analyze embeddings using different techniques
- Compare runs
- Generate visualizations (UMAP, etc)
- Produce structured reports

## project code tree

```
ComfyUI-EmbeddingPipelineAnalytics/
├── core
│   ├── analyzers.py
│   ├── data_store.py
│   ├── __init__.py
│   └── pipeline_types
│       ├── base.py
│       ├── hunyuanvideo.py
│       ├── __init__.py
│       └── __pycache__
├── example_workflows
│   └── example_workflow.json
├── __init__.py
├── LICENSE
├── nodes.py
├── README.md
├── requirements.txt
└── tests
    ├── conftest.py
    ├── test_hunyuanvideo.py
    └── test_workflow.py
```
