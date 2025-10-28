[中文](./LVLM_cn.md) |
[English](./LVLM_en.md)
**LVLM** stands for **Large Vision-Language Model**.  
It refers to a class of AI models capable of **understanding both visual information (e.g., images, videos)** and **language information (e.g., text, speech)** simultaneously.

---

## I. Definition
LVLMs are an extension of **Large Language Models (LLMs)** that incorporate visual modalities as inputs. This allows the model not only to *read text* but also to *interpret images*.  
A typical architecture includes:

> **Vision Encoder + Large Language Model (LLM) + Fusion Module (Connector / Projector)**

---

## II. Working Principle
1. **Visual Encoding:** Visual models (such as CLIP, ViT, ResNet, or SigLIP) extract image features.  
2. **Feature Projection:** A Projector maps these visual features into the semantic space understandable by the LLM.  
3. **Language Generation:** The LLM (e.g., LLaMA, Qwen, GPT) generates responses based on both visual semantics and textual context.

---

## III. Common Applications
- Image understanding and captioning  
- Visual Question Answering (VQA)  
- Cross-modal retrieval (image–text matching)  
- Text-guided image analysis  
- Document/Table visual understanding  
- Multimodal RAG (joint text-image retrieval and generation)

---

## IV. Representative Models

| Model Name | Organization | Base LLM | Key Features |
|-------------|---------------|-----------|---------------|
| GPT-4V | OpenAI | GPT-4 | Supports visual inputs and multimodal conversations |
| Gemini 1.5 Pro | Google DeepMind | Gemini | Handles image, audio, and video understanding |
| LLaVA / LLaVA-Next | UC Berkeley | LLaMA | Open-source, highly efficient training |
| Qwen-VL / Qwen2-VL | Alibaba | Qwen | Strong Chinese language understanding |
| MiniGPT-4 / mPLUG-Owl | Huawei, etc. | Vicuna / LLaMA | Lightweight, locally deployable |

---

## V. Differences Between LLM and LVLM

| Comparison Item | LLM | LVLM |
|------------------|------|------|
| Input | Text only | Image + Text |
| Modality | Single-modal | Multimodal |
| Capability | Language understanding & generation | Joint visual–language understanding |
| Applications | Chat, coding, text generation | Visual Q&A, image analysis, document parsing |

---
