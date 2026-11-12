"""Unsupervised analysis of clip embeddings: PCA/LDA, K-Means/agglomerative clustering, autoencoder anomaly detection.

The entry point for analysis/unsupervised.py (the desktop's Features & probes page runs it as a subprocess).

    python analyse_embeddings.py --embeddings run/embed_output_x/x_cls_embeddings.parquet --out run/unsupervised
"""
import sys

from analysis.unsupervised import build_parser, main

__all__ = ['build_parser', 'main']

if __name__ == '__main__':
    sys.exit(main())
