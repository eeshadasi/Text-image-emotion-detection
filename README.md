# Image Captioning and Emotion Detection with BLIP and Gradio

## Overview

This project uses BLIP (Bootstrapped Language Image Pre-training) for image captioning and multiple emotion detection models to analyze the generated captions. The results are displayed through a Gradio interface that allows users to upload images, generate captions, and detect emotions from those captions.

## Features

- **Image Captioning**: Uses the BLIP model to generate captions for uploaded images.
- **Emotion Detection**: Analyzes the generated captions using multiple emotion detection models.
- **Gradio Interface**: Provides a user-friendly web interface to upload images and view results.

## Prerequisites

- Python 3.7 or higher
- Required Python packages:
  - `transformers`
  - `torch`
  - `PIL` (Pillow)
  - `gradio`
  - `requests`
  - `io`

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
