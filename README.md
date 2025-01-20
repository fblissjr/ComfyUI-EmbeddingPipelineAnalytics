# ComfyUI-EmbeddingPipelineAnalytics

## Overview

This repo is to capture end-to-end data, metadata, and embeddings for ComfyUI workflows, specifically HunyuanVideo to start.

## Goals

- Creating more formalized datasets for open source projects and teams, for downstream finetuning or analysis
- Giving more powerful analysis tools and data management tooling to non-developers in the community
- Providing a feedback mechanism for analyzing embeddings vs. inputs

## Features

As this will be maintained based on my own free time, the core will focus on the following:

*Generic capture node that can:*

- Capture embeddings and run metadata and data inputs / outputs from ComfyUI workflows (parameters, prompt, guidance, sampling, etc)
- Store data in a structured format for analysis (parquet, jsonl)

*Analysis node that can:*

- Analyze embeddings using different techniques
- Compare runs
- Generate visualizations (UMAP, etc)
- Produce structured reports
