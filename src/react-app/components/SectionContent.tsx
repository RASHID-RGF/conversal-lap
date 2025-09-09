import { useGuide } from '@/react-app/contexts/GuideContext';
import { CheckCircle, Circle, ArrowRight, ArrowLeft, ExternalLink, Copy, Check } from 'lucide-react';
import { useState } from 'react';
import { useNavigate } from 'react-router';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

// Section content data
const sectionContent = {
  overview: {
    title: "Overview & Introduction to Conversational AI",
    content: `
# Welcome to ConversaLab

Building an advanced conversational AI system like ChatGPT is one of the most exciting and challenging projects in modern technology. This comprehensive guide will walk you through every aspect of the process, from understanding the fundamental components to deploying a production-ready system.

## What You'll Learn

By the end of this guide, you'll understand:

- **Core Architecture**: How large language models, tokenizers, and inference systems work together
- **Data Pipeline**: Best practices for collecting, cleaning, and managing training data
- **Model Selection**: Choosing between open-source options like LLaMA, Mistral, and Falcon
- **Infrastructure**: Hardware requirements and cost-effective cloud deployment strategies
- **Fine-tuning**: Customizing models for specific domains and use cases
- **Safety & Ethics**: Implementing robust moderation and alignment techniques
- **Production**: Building scalable APIs and maintaining systems in production

## The Challenge Ahead

Creating a conversational AI system requires expertise across multiple domains:

- **Machine Learning**: Understanding transformer architectures and training dynamics
- **Data Engineering**: Processing terabytes of text data efficiently and safely
- **Infrastructure**: Managing GPU clusters and distributed training
- **Software Engineering**: Building robust, scalable APIs and user interfaces
- **Ethics & Safety**: Ensuring responsible AI development and deployment

## Prerequisites

While this guide is comprehensive, some background knowledge will be helpful:

- Basic understanding of machine learning concepts
- Familiarity with Python programming
- Knowledge of cloud computing concepts
- Understanding of web APIs and software architecture

Don't worry if you're not an expert in all these areas - we'll provide explanations and resources throughout the guide.

## Project Scope & Timeline

Building a production-ready conversational AI is a significant undertaking:

- **Small Team (3-5 people)**: 6-12 months for MVP
- **Medium Team (10-20 people)**: 3-6 months for MVP
- **Large Team (50+ people)**: 1-3 months for MVP

Budget considerations range from $10K for experiments to $1M+ for production systems.

## Getting Started

Ready to begin? The next section covers the core components that make conversational AI systems work. Click "Next Section" when you're ready to dive deeper.
    `
  },
  "core-components": {
    title: "Core Components of Conversational AI",
    content: `
# Core Components of Conversational AI

Understanding the fundamental building blocks is crucial before diving into implementation. Modern conversational AI systems consist of several interconnected components working together to process and generate human-like text.

## 1. Large Language Models (LLMs)

### What are LLMs?
Large Language Models are neural networks trained on vast amounts of text data to understand and generate human language. They use the **Transformer architecture**, which revolutionized natural language processing.

### Key Characteristics:
- **Scale**: Billions to trillions of parameters
- **Training**: Self-supervised learning on massive text corpora  
- **Capability**: Understanding context, reasoning, and generating coherent responses

### Popular LLM Architectures:

\`\`\`
GPT (Generative Pre-trained Transformer)
├── GPT-3.5: 175B parameters
├── GPT-4: ~1.76T parameters (estimated)
└── GPT-4o: Optimized for speed and efficiency

LLaMA (Large Language Model Meta AI)
├── LLaMA 2: 7B, 13B, 70B parameter variants
└── Code Llama: Specialized for code generation

Mistral
├── Mistral 7B: Efficient 7B parameter model
└── Mixtral 8x7B: Mixture of experts architecture
\`\`\`

## 2. Tokenization System

### Purpose
Tokenization converts human text into numerical representations that neural networks can process.

### Common Tokenization Methods:

**Byte Pair Encoding (BPE)**
- Splits text into subword units
- Balances vocabulary size with representation efficiency
- Used by GPT models

**SentencePiece**
- Language-agnostic tokenization
- Handles multiple languages and scripts
- Used by T5 and many multilingual models

### Example Tokenization:
\`\`\`python
# Input text
text = "Hello, how are you today?"

# BPE tokens (approximate)
tokens = ["Hello", ",", "how", "are", "you", "today", "?"]
token_ids = [15496, 11, 703, 389, 345, 1909, 30]
\`\`\`

## 3. Inference Engine

### Real-time Processing
The inference engine handles:
- **Input Processing**: Converting user queries to tokens
- **Model Execution**: Running the neural network forward pass
- **Output Generation**: Converting model outputs back to text
- **Optimization**: Batching, caching, and acceleration

### Key Optimization Techniques:

**KV-Caching**
- Stores previously computed key-value pairs
- Reduces computation for long conversations
- Critical for chat applications

**Quantization**
- Reduces model precision (FP32 → FP16 → INT8)
- Decreases memory usage and increases speed
- Trade-off between efficiency and quality

**Speculative Decoding**
- Uses smaller models to predict future tokens
- Verifies predictions with the main model
- Can significantly speed up generation

## 4. Context Management

### Conversation History
Managing context across multiple turns:

\`\`\`python
class ConversationContext:
    def __init__(self, max_tokens=4096):
        self.max_tokens = max_tokens
        self.history = []
    
    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})
        self._truncate_if_needed()
    
    def _truncate_if_needed(self):
        # Implement token counting and truncation logic
        pass
\`\`\`

### Memory Management Strategies:
- **Sliding Window**: Keep only recent messages
- **Summarization**: Compress older conversation parts
- **Hierarchical Memory**: Different retention policies for different content types

## 5. Safety & Alignment Layer

### Content Filtering
Multiple layers of safety:

1. **Input Filtering**: Screen user inputs for harmful content
2. **Output Filtering**: Check generated responses before serving
3. **Constitutional AI**: Train models to follow helpful, harmless principles
4. **Red Team Testing**: Adversarial testing for edge cases

### Example Safety Pipeline:
\`\`\`python
def safety_check(text):
    # Multiple safety checks
    if contains_harmful_content(text):
        return False, "Content policy violation"
    
    if contains_personal_info(text):
        return False, "Privacy concern"
    
    if toxicity_score(text) > THRESHOLD:
        return False, "High toxicity detected"
    
    return True, "Safe"
\`\`\`

## Next Steps

Now that you understand the core components, the next section will dive into data management - how to collect, process, and handle the massive datasets needed to train these systems effectively.
    `
  },
  "data-management": {
    title: "Data Management & Ethics",
    content: `
# Data Management for Conversational AI

Data is the foundation of any successful conversational AI system. The quality, diversity, and ethical handling of training data directly impact model performance, safety, and societal impact.

## 1. Data Collection Strategies

### Web Scraping & Crawling

**Common Data Sources:**
\`\`\`
Web Content
├── Wikipedia: High-quality, factual content
├── News Articles: Current events and journalism
├── Forums: Reddit, Stack Overflow, specialized communities
├── Books & Literature: Project Gutenberg, open libraries
└── Academic Papers: arXiv, PubMed, research repositories
\`\`\`

**Scraping Best Practices:**
\`\`\`python
import asyncio
import aiohttp
from bs4 import BeautifulSoup

class EthicalScraper:
    def __init__(self, rate_limit=1.0):
        self.rate_limit = rate_limit
        self.session = None
    
    async def scrape_url(self, url):
        # Respect robots.txt
        if not self.check_robots_txt(url):
            return None
        
        # Rate limiting
        await asyncio.sleep(self.rate_limit)
        
        # Scrape with proper headers
        headers = {
            'User-Agent': 'Research Bot 1.0 (contact@example.com)',
            'Accept': 'text/html,application/xhtml+xml'
        }
        
        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                return await response.text()
        
        return None
\`\`\`

### Licensed Datasets

**High-Quality Sources:**
- **Common Crawl**: Web crawl data (free, but requires filtering)
- **OpenWebText**: Open-source version of WebText
- **C4 (Colossal Clean Crawled Corpus)**: Cleaned web text
- **The Pile**: 800GB of diverse text data
- **BookCorpus**: Collection of books (licensing required)

## 2. Data Preprocessing Pipeline

### Text Cleaning & Normalization

**Essential Cleaning Steps:**
\`\`\`python
import re
from typing import List

class TextCleaner:
    def __init__(self):
        self.patterns = {
            'email': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',
            'phone': r'\d{3}[-.]?\d{3}[-.]?\d{4}',
            'ssn': r'\d{3}-?\d{2}-?\d{4}',
            'credit_card': r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}'
        }
    
    def remove_pii(self, text: str) -> str:
        """Remove personally identifiable information"""
        for pattern_name, pattern in self.patterns.items():
            text = re.sub(pattern, f'[{pattern_name.upper()}_REDACTED]', text)
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and line breaks"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Fix line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()
\`\`\`

## 3. Ethical Data Handling

### Privacy Protection

**Key Principles:**
1. **Data Minimization**: Collect only necessary data
2. **Purpose Limitation**: Use data only for stated purposes
3. **Transparency**: Clear data usage policies
4. **User Control**: Options for data deletion/correction

### Bias Mitigation

**Sources of Bias:**
- **Demographic Bias**: Underrepresentation of certain groups
- **Temporal Bias**: Outdated information or changing social norms
- **Source Bias**: Skew toward certain websites or publications
- **Annotation Bias**: Human labeler preferences and backgrounds

## Next Steps

With a solid foundation in data management, you're ready to explore the open-source frameworks and tools that will help you build your conversational AI system.
    `
  },
  "frameworks": {
    title: "Open Source Frameworks & Tools",
    content: `
# Open Source Frameworks & Tools

The open-source ecosystem provides powerful tools and pre-trained models that dramatically accelerate conversational AI development. This section covers the most important frameworks and libraries.

## 1. Hugging Face Ecosystem

### Transformers Library

The de facto standard for working with transformer models:

\`\`\`python
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline
)

# Quick start with a pre-trained model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Simple conversation pipeline
chatbot = pipeline(
    "conversational",
    model=model,
    tokenizer=tokenizer
)
\`\`\`

### Key Features:
- **50,000+ Models**: Pre-trained models for every use case
- **Automatic Downloads**: Seamless model and tokenizer loading
- **Multiple Backends**: PyTorch, TensorFlow, JAX support
- **Production Ready**: Optimized inference and serving

## 2. LLaMA Family Models

### LLaMA 2 Overview

Meta's LLaMA 2 offers state-of-the-art performance with permissive licensing:

**Model Variants:**
- **LLaMA 2-7B**: Best for resource-constrained environments
- **LLaMA 2-13B**: Balanced performance and efficiency  
- **LLaMA 2-70B**: Top performance for complex tasks

**Chat Variants:** Fine-tuned for conversations with safety measures

\`\`\`python
# Loading LLaMA 2 with Transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
\`\`\`

## 3. Mistral AI Models

### Mistral 7B

Highly efficient 7B parameter model with excellent performance:

**Key Advantages:**
- Fast inference
- High quality outputs
- Commercial-friendly license
- Multi-language support

## 4. Training Frameworks

### LoRA (Low-Rank Adaptation)

\`\`\`python
from peft import LoraConfig, get_peft_model

# LoRA configuration
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

# Apply LoRA to model
model = get_peft_model(base_model, lora_config)
\`\`\`

### QLoRA (Quantized LoRA)

\`\`\`python
from transformers import BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
\`\`\`

## Next Steps

With these frameworks and tools, you're equipped to start building. The next section covers the infrastructure and compute resources needed effectively.
    `
  },
  "infrastructure": {
    title: "Infrastructure & Computing Resources",
    content: `
# Infrastructure & Computing Resources

Building conversational AI requires significant computational resources for both training and inference. This section covers hardware requirements, cloud options, and cost-effective strategies.

## 1. Hardware Requirements Overview

### Training Infrastructure

**Minimum Requirements (Fine-tuning 7B model):**
\`\`\`
GPU Memory: 24GB+ (RTX 4090, A6000)
System RAM: 64GB+
Storage: 2TB NVMe SSD
Network: 10 Gbps for multi-GPU setups
\`\`\`

**Recommended Setup (Training 7B from scratch):**
\`\`\`
GPU Cluster: 8x A100 (80GB) or H100
System RAM: 512GB+
Storage: 10TB+ high-speed storage
Network: InfiniBand for GPU communication
\`\`\`

**Large Scale (70B+ models):**
\`\`\`
GPU Cluster: 64+ A100/H100 GPUs
System RAM: 2TB+
Storage: 100TB+ distributed storage
Network: High-bandwidth InfiniBand fabric
\`\`\`

## 2. Cloud Provider Comparison

### Amazon Web Services (AWS)

**GPU Instances:**
\`\`\`python
# Cost analysis for AWS EC2 instances (hourly rates)
aws_instances = {
    'p4d.24xlarge': {
        'gpus': '8x A100 (40GB)',
        'price_per_hour': 32.77,
        'use_case': 'Large model training'
    },
    'p4de.24xlarge': {
        'gpus': '8x A100 (80GB)', 
        'price_per_hour': 40.96,
        'use_case': 'Very large model training'
    },
    'g5.xlarge': {
        'gpus': '1x A10G (24GB)',
        'price_per_hour': 1.006,
        'use_case': 'Inference, small model training'
    }
}
\`\`\`

**AWS Services for AI:**
- **SageMaker**: Managed ML platform
- **Bedrock**: Managed foundation model API
- **EKS**: Kubernetes for containerized deployments
- **S3**: Scalable data storage

### Google Cloud Platform (GCP)

**GPU Options:**
- **A100 instances**: High-performance training
- **T4 instances**: Cost-effective inference
- **TPUs**: Google's custom AI accelerators

**GCP AI Services:**
- **Vertex AI**: End-to-end ML platform
- **GKE**: Kubernetes with GPU support
- **Cloud Storage**: Scalable data storage

### Microsoft Azure

**GPU Instances:**
- **NC series**: NVIDIA GPU instances
- **ND series**: High-end AI training

**Azure AI Services:**
- **Azure Machine Learning**: Comprehensive ML platform
- **Azure OpenAI Service**: Hosted GPT models
- **AKS**: Kubernetes with GPU support

## 3. Cost Optimization Strategies

### Spot Instances & Preemptible VMs

\`\`\`python
class SpotInstanceManager:
    def __init__(self, region='us-east-1'):
        self.ec2 = boto3.client('ec2', region_name=region)
    
    def get_spot_prices(self, instance_types):
        """Get current spot prices for instance types"""
        response = self.ec2.describe_spot_price_history(
            InstanceTypes=instance_types,
            ProductDescriptions=['Linux/UNIX'],
            MaxResults=len(instance_types)
        )
        
        prices = {}
        for price in response['SpotPrices']:
            prices[price['InstanceType']] = float(price['SpotPrice'])
        
        return prices

# Potential savings: 50-90% compared to on-demand
\`\`\`

### Reserved Instances & Committed Use

For predictable workloads, reserved instances can provide 25-40% savings over on-demand pricing.

## 4. Container Orchestration

### Kubernetes for AI Workloads

\`\`\`yaml
# GPU-enabled Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llama-inference
  template:
    metadata:
      labels:
        app: llama-inference
    spec:
      containers:
      - name: llama-server
        image: your-registry/llama-server:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
        env:
        - name: MODEL_PATH
          value: "/models/llama-2-7b"
\`\`\`

## Next Steps

With your infrastructure foundation in place, you're ready to learn about fine-tuning techniques that will customize your models for specific use cases.
    `
  },
  "fine-tuning": {
    title: "Fine-tuning & Customization",
    content: `
# Fine-tuning & Customization

Fine-tuning adapts pre-trained models to specific domains, tasks, or conversation styles. This section covers various fine-tuning techniques, from lightweight approaches to full model retraining.

## 1. Fine-tuning Approaches Overview

### Technique Comparison

| Method | Memory Usage | Training Time | Quality | Use Case |
|--------|--------------|---------------|---------|----------|
| **LoRA** | Very Low | Hours | Excellent | Most scenarios |
| **QLoRA** | Ultra Low | Hours | Excellent | Limited resources |
| **Full Fine-tuning** | Very High | Days/Weeks | Best | Large budgets |
| **RLHF** | High | Days | Best | Alignment tasks |

## 2. LoRA (Low-Rank Adaptation)

### Why LoRA Works

LoRA freezes the original model weights and injects trainable rank decomposition matrices into each layer. This dramatically reduces the number of trainable parameters while maintaining performance.

### LoRA Implementation

\`\`\`python
from peft import LoraConfig, get_peft_model
import torch

# LoRA configuration
lora_config = LoraConfig(
    r=16,  # Rank - controls adapter size
    lora_alpha=32,  # LoRA scaling parameter
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Add LoRA adapters
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06
\`\`\`

### Training with LoRA

\`\`\`python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./lora_results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=25,
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# Start training
trainer.train()
\`\`\`

## 3. QLoRA (Quantized LoRA)

### Memory-Efficient Training

QLoRA combines 4-bit quantization with LoRA, enabling fine-tuning of large models on consumer hardware.

\`\`\`python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# Add LoRA
model = get_peft_model(model, lora_config)
\`\`\`

**Memory Usage Comparison:**
- **Full Fine-tuning**: ~52GB for 13B model
- **LoRA**: ~14GB for 13B model  
- **QLoRA**: ~7GB for 13B model

## 4. Domain-Specific Fine-tuning

### Customer Support Bot

\`\`\`python
# Example conversation format for customer support
conversation_data = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful customer support agent."},
            {"role": "user", "content": "My order hasn't arrived yet."},
            {"role": "assistant", "content": "I understand your concern. Let me help track your order. Could you provide your order number?"}
        ]
    }
]

# Convert to training format
def format_conversation(conversation):
    formatted = "<s>"
    for message in conversation["messages"]:
        if message["role"] == "user":
            formatted += f"[INST] {message['content']} [/INST]"
        elif message["role"] == "assistant":
            formatted += f" {message['content']} </s><s>"
    return formatted.rstrip("<s>")
\`\`\`

### Code Assistant

\`\`\`python
# Code-specific conversation format
code_conversations = [
    {
        "messages": [
            {"role": "user", "content": "Write a Python function to calculate factorial"},
            {"role": "assistant", "content": "Here's a Python function to calculate factorial:\\n\\n```python\\ndef factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n - 1)\\n```\\n\\nThis recursive function calculates the factorial by multiplying n with factorial(n-1)."}
        ]
    }
]
\`\`\`

## 5. Evaluation & Monitoring

### Quality Metrics

\`\`\`python
def evaluate_model(model, test_dataset):
    metrics = {
        "perplexity": calculate_perplexity(model, test_dataset),
        "bleu_score": calculate_bleu(model, test_dataset),
        "rouge_score": calculate_rouge(model, test_dataset),
        "human_eval": conduct_human_evaluation(model, test_dataset)
    }
    return metrics

def calculate_perplexity(model, dataset):
    total_loss = 0
    total_tokens = 0
    
    for batch in dataset:
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item() * batch["input_ids"].size(1)
            total_tokens += batch["input_ids"].size(1)
    
    return torch.exp(torch.tensor(total_loss / total_tokens))
\`\`\`

### A/B Testing Framework

\`\`\`python
class ABTestFramework:
    def __init__(self, model_a, model_b):
        self.model_a = model_a
        self.model_b = model_b
        self.results = []
    
    def run_comparison(self, test_prompts, num_users=1000):
        for prompt in test_prompts:
            # Generate responses from both models
            response_a = self.model_a.generate(prompt)
            response_b = self.model_b.generate(prompt)
            
            # Collect user preferences
            preference = self.collect_user_preference(
                prompt, response_a, response_b
            )
            
            self.results.append({
                "prompt": prompt,
                "response_a": response_a,
                "response_b": response_b,
                "preference": preference
            })
    
    def analyze_results(self):
        a_wins = sum(1 for r in self.results if r["preference"] == "a")
        b_wins = sum(1 for r in self.results if r["preference"] == "b")
        ties = sum(1 for r in self.results if r["preference"] == "tie")
        
        return {
            "model_a_win_rate": a_wins / len(self.results),
            "model_b_win_rate": b_wins / len(self.results),
            "tie_rate": ties / len(self.results)
        }
\`\`\`

## Next Steps

With fine-tuning techniques mastered, the next section covers safety and ethics - crucial considerations for responsible AI deployment.
    `
  },
  "safety": {
    title: "Safety & Ethics",
    content: `
# Safety & Ethics in Conversational AI

Ensuring AI safety and ethical behavior is crucial for responsible deployment. This section covers moderation techniques, bias prevention, and alignment strategies.

## 1. Content Moderation Pipeline

### Multi-Layer Safety Architecture

\`\`\`python
class SafetyPipeline:
    def __init__(self):
        self.input_filters = [
            self.check_harmful_content,
            self.check_personal_info,
            self.check_inappropriate_requests
        ]
        self.output_filters = [
            self.check_generated_content,
            self.check_factual_accuracy,
            self.check_bias_indicators
        ]
    
    def process_input(self, user_input):
        """Filter user input before processing"""
        for filter_func in self.input_filters:
            is_safe, reason = filter_func(user_input)
            if not is_safe:
                return False, f"Input rejected: {reason}"
        return True, "Input approved"
    
    def process_output(self, generated_response):
        """Filter generated response before serving"""
        for filter_func in self.output_filters:
            is_safe, reason = filter_func(generated_response)
            if not is_safe:
                return False, f"Response blocked: {reason}"
        return True, "Response approved"
    
    def check_harmful_content(self, text):
        """Check for harmful or toxic content"""
        # Use services like Perspective API or custom models
        toxicity_score = self.get_toxicity_score(text)
        if toxicity_score > 0.7:
            return False, "High toxicity detected"
        return True, "Content acceptable"
    
    def check_personal_info(self, text):
        """Check for personally identifiable information"""
        pii_patterns = {
            'email': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',
            'phone': r'\d{3}[-.]?\d{3}[-.]?\d{4}',
            'ssn': r'\d{3}-?\d{2}-?\d{4}'
        }
        
        for pii_type, pattern in pii_patterns.items():
            if re.search(pattern, text):
                return False, f"Contains {pii_type}"
        
        return True, "No PII detected"
\`\`\`

### Real-time Toxicity Detection

\`\`\`python
import requests

class ToxicityDetector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.perspective_url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
    
    def analyze_toxicity(self, text):
        """Use Google's Perspective API for toxicity detection"""
        data = {
            'comment': {'text': text},
            'requestedAttributes': {
                'TOXICITY': {},
                'SEVERE_TOXICITY': {},
                'IDENTITY_ATTACK': {},
                'INSULT': {},
                'PROFANITY': {},
                'THREAT': {}
            }
        }
        
        response = requests.post(
            f"{self.perspective_url}?key={self.api_key}",
            json=data
        )
        
        if response.status_code == 200:
            scores = response.json()['attributeScores']
            return {
                attr: scores[attr]['summaryScore']['value']
                for attr in scores
            }
        else:
            return None
    
    def is_safe(self, text, threshold=0.7):
        """Determine if text is safe based on toxicity scores"""
        scores = self.analyze_toxicity(text)
        if not scores:
            return True  # Default to safe if API fails
        
        for attr, score in scores.items():
            if score > threshold:
                return False
        
        return True
\`\`\`

## 2. Bias Detection and Mitigation

### Demographic Bias Testing

\`\`\`python
class BiasDetector:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Test cases for different demographic groups
        self.bias_test_cases = {
            'gender': [
                "The engineer {pronoun} is working on the project.",
                "The nurse {pronoun} is caring for patients.",
                "The CEO {pronoun} announced new policies."
            ],
            'race': [
                "The person from {country} is very {adjective}.",
                "{name} is a talented {profession}."
            ],
            'religion': [
                "People who practice {religion} are often {adjective}."
            ]
        }
    
    def test_gender_bias(self):
        """Test for gender bias in model responses"""
        results = {}
        
        for template in self.bias_test_cases['gender']:
            male_version = template.format(pronoun="he")
            female_version = template.format(pronoun="she")
            
            male_response = self.generate_response(male_version)
            female_response = self.generate_response(female_version)
            
            # Analyze sentiment and content differences
            bias_score = self.compare_responses(male_response, female_response)
            
            results[template] = {
                'male_response': male_response,
                'female_response': female_response,
                'bias_score': bias_score
            }
        
        return results
    
    def compare_responses(self, response1, response2):
        """Compare two responses for bias indicators"""
        # Use sentiment analysis, keyword comparison, etc.
        sentiment1 = self.analyze_sentiment(response1)
        sentiment2 = self.analyze_sentiment(response2)
        
        # Calculate bias score (0 = no bias, 1 = maximum bias)
        sentiment_diff = abs(sentiment1 - sentiment2)
        return min(sentiment_diff, 1.0)
    
    def generate_response(self, prompt):
        """Generate response from model"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=100)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()
\`\`\`

## Next Steps

With safety and ethics foundations in place, the next section covers building production APIs and integrating your conversational AI into applications.
    `
  },
  "api-integration": {
    title: "API & Integration",
    content: `
# API & Integration

Building production-ready APIs and integrating conversational AI into applications requires careful consideration of performance, scalability, and user experience.

## 1. RESTful API Design

### Core API Structure

\`\`\`python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime

app = FastAPI(title="ConversaLab API", version="1.0.0")

# Request/Response Models
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    id: str
    choices: List[dict]
    usage: dict
    conversation_id: str
    created: datetime

# API Endpoints
@app.post("/v1/chat/completions", response_model=ChatResponse)
async def create_chat_completion(request: ChatRequest):
    """Create a chat completion"""
    try:
        return await chat_completion.create_completion(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": "conversalab-7b",
                "object": "model",
                "created": 1677610602,
                "owned_by": "conversalab"
            }
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}
\`\`\`

### Authentication & Rate Limiting

\`\`\`python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import hashlib
import time
from collections import defaultdict

security = HTTPBearer()

class APIKeyManager:
    def __init__(self):
        self.api_keys = {}  # In production, use a database
        self.rate_limits = defaultdict(list)  # Track request times
        
    def create_api_key(self, user_id: str, tier: str = "basic") -> str:
        """Create a new API key"""
        key = f"cl-{hashlib.sha256(f'{user_id}{time.time()}'.encode()).hexdigest()[:32]}"
        
        self.api_keys[key] = {
            "user_id": user_id,
            "tier": tier,
            "created": datetime.now(),
            "last_used": None,
            "requests_count": 0
        }
        
        return key
    
    def validate_api_key(self, api_key: str) -> dict:
        """Validate API key and return user info"""
        if api_key not in self.api_keys:
            return None
        
        key_info = self.api_keys[api_key]
        key_info["last_used"] = datetime.now()
        key_info["requests_count"] += 1
        
        return key_info
    
    def check_rate_limit(self, api_key: str, tier: str) -> bool:
        """Check if request is within rate limits"""
        current_time = time.time()
        
        # Rate limits by tier (requests per minute)
        limits = {
            "basic": 10,
            "pro": 100,
            "enterprise": 1000
        }
        
        limit = limits.get(tier, 10)
        
        # Clean old requests (older than 1 minute)
        self.rate_limits[api_key] = [
            req_time for req_time in self.rate_limits[api_key]
            if current_time - req_time < 60
        ]
        
        # Check if under limit
        if len(self.rate_limits[api_key]) >= limit:
            return False
        
        # Add current request
        self.rate_limits[api_key].append(current_time)
        return True

api_key_manager = APIKeyManager()

async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to validate API key"""
    api_key = credentials.credentials
    
    # Validate API key
    key_info = api_key_manager.validate_api_key(api_key)
    if not key_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    # Check rate limits
    if not api_key_manager.check_rate_limit(api_key, key_info["tier"]):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    return key_info
\`\`\`

## 2. WebSocket Integration

### Real-time Chat Implementation

\`\`\`python
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.conversations = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Validate message format
            if "content" not in message_data:
                await websocket.send_text(json.dumps({
                    "error": "Missing content field"
                }))
                continue
            
            # Process with AI model
            user_message = message_data["content"]
            
            # Generate response
            ai_response = await chat_completion.generate_response(user_message)
            
            # Send response back
            response_data = {
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().isoformat(),
                "conversation_id": conversation_id
            }
            
            await manager.send_personal_message(
                json.dumps(response_data),
                websocket
            )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
\`\`\`

## 3. SDK and Client Libraries

### Python SDK

\`\`\`python
import requests
import json
from typing import List, Optional, Iterator

class ConversaLabClient:
    def __init__(self, api_key: str, base_url: str = "https://api.conversalab.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def chat_completion(
        self,
        messages: List[dict],
        max_tokens: Optional[int] = 512,
        temperature: Optional[float] = 0.7,
        stream: bool = False
    ) -> dict:
        """Create a chat completion"""
        
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        if stream:
            return self._stream_completion(payload)
        else:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            )
            response.raise_for_status()
            return response.json()
    
    def list_models(self) -> dict:
        """List available models"""
        response = self.session.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json()

# Usage example
client = ConversaLabClient(api_key="your-api-key")

# Simple completion
response = client.chat_completion([
    {"role": "user", "content": "Hello, how are you?"}
])

print(response["choices"][0]["message"]["content"])
\`\`\`

## Next Steps

With your API and integration foundation complete, the final section covers best practices for scaling and maintaining your conversational AI system in production.
    `
  },
  "scaling": {
    title: "Scaling & Maintenance",
    content: `
# Scaling & Maintenance

Maintaining a production conversational AI system requires careful attention to performance, reliability, monitoring, and continuous improvement.

## 1. Performance Optimization

### Model Optimization Techniques

\`\`\`python
import torch
from torch import nn
from transformers import AutoModelForCausalLM
import onnxruntime as ort

class ModelOptimizer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.optimized_models = {}
    
    def quantize_model(self, quantization_type="dynamic"):
        """Quantize model for faster inference"""
        if quantization_type == "dynamic":
            # Dynamic quantization
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},  # Specify layers to quantize
                dtype=torch.qint8
            )
            return quantized_model
        
        elif quantization_type == "static":
            # Static quantization (requires calibration dataset)
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            model.eval()
            
            # Prepare for static quantization
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            
            # Calibrate with sample data (you need to provide this)
            self.calibrate_model(model)
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(model, inplace=True)
            return quantized_model

# Usage
optimizer = ModelOptimizer("meta-llama/Llama-2-7b-chat-hf")

# Quantize model
quantized_model = optimizer.quantize_model("dynamic")
\`\`\`

### Caching Strategies

\`\`\`python
import redis
import hashlib
import json
from typing import Optional
import pickle

class ResponseCache:
    def __init__(self, redis_host='localhost', redis_port=6379, ttl=3600):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
        self.ttl = ttl  # Time to live in seconds
    
    def _create_cache_key(self, messages, temperature, max_tokens):
        """Create a unique cache key for the request"""
        cache_input = {
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        # Create hash of the input
        cache_str = json.dumps(cache_input, sort_keys=True)
        cache_key = hashlib.md5(cache_str.encode()).hexdigest()
        return f"response_cache:{cache_key}"
    
    def get_cached_response(self, messages, temperature, max_tokens) -> Optional[str]:
        """Get cached response if available"""
        cache_key = self._create_cache_key(messages, temperature, max_tokens)
        
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            print(f"Cache retrieval error: {e}")
        
        return None
    
    def cache_response(self, messages, temperature, max_tokens, response):
        """Cache the response"""
        cache_key = self._create_cache_key(messages, temperature, max_tokens)
        
        try:
            # Serialize and cache the response
            serialized_response = pickle.dumps(response)
            self.redis_client.setex(cache_key, self.ttl, serialized_response)
        except Exception as e:
            print(f"Cache storage error: {e}")

# Usage
cache = ResponseCache(ttl=1800)  # 30 minute cache
\`\`\`

## 2. Auto-scaling Infrastructure

### Kubernetes Auto-scaling

\`\`\`yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: conversalab-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: conversalab-api
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
\`\`\`

## 3. Monitoring & Observability

### Comprehensive Monitoring Stack

\`\`\`python
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import logging
from functools import wraps

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')
MODEL_INFERENCE_TIME = Histogram('model_inference_seconds', 'Model inference time')

class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()
        
        # Start Prometheus metrics server
        start_http_server(8000)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def track_request(self, method, endpoint):
        """Decorator to track API requests"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                status = 'success'
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = 'error'
                    self.logger.error(f"Request failed: {e}")
                    raise
                finally:
                    duration = time.time() - start_time
                    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
                    REQUEST_DURATION.observe(duration)
                    
                    self.logger.info(f"{method} {endpoint} - {status} - {duration:.3f}s")
            
            return wrapper
        return decorator

# Usage in API endpoints
metrics = MetricsCollector()

@app.post("/v1/chat/completions")
@metrics.track_request("POST", "/v1/chat/completions")
async def create_chat_completion(request: ChatRequest):
    ACTIVE_CONNECTIONS.inc()
    
    try:
        # Your API logic here
        return await chat_completion.create_completion(request)
    finally:
        ACTIVE_CONNECTIONS.dec()
\`\`\`

## Conclusion

Congratulations! You've completed the comprehensive ConversaLab guide to building advanced conversational AI systems. You now have the knowledge and tools to:

- **Design robust architectures** with proper component separation
- **Manage training data** ethically and effectively
- **Leverage open-source frameworks** for rapid development
- **Deploy on scalable infrastructure** with cost optimization
- **Fine-tune models** for specific domains and use cases
- **Implement safety measures** and ethical AI practices
- **Build production APIs** with proper authentication and rate limiting
- **Scale and maintain** systems for long-term success

## Next Steps

1. **Start Small**: Begin with a proof-of-concept using existing models
2. **Iterate Quickly**: Use LoRA/QLoRA for rapid experimentation
3. **Measure Everything**: Implement comprehensive monitoring from day one
4. **Safety First**: Build safety and ethics into your system architecture
5. **Scale Gradually**: Grow your infrastructure as demand increases

The field of conversational AI is rapidly evolving. Stay updated with the latest research, maintain high safety standards, and always consider the societal impact of your AI systems.

Happy building! 🚀
    `
  }
};

export default function SectionContent() {
  const { sections, currentSection, markCompleted } = useGuide();
  const navigate = useNavigate();
  const [copiedCode, setCopiedCode] = useState<string | null>(null);
  
  const currentSectionData = sections.find(s => s.id === currentSection);
  const content = sectionContent[currentSection as keyof typeof sectionContent];
  
  const currentIndex = sections.findIndex(s => s.id === currentSection);
  const previousSection = currentIndex > 0 ? sections[currentIndex - 1] : null;
  const nextSection = currentIndex < sections.length - 1 ? sections[currentIndex + 1] : null;

  const copyToClipboard = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedCode(id);
      setTimeout(() => setCopiedCode(null), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const handleComplete = () => {
    if (currentSectionData) {
      markCompleted(currentSectionData.id);
    }
  };

  const goToSection = (sectionId: string) => {
    navigate(`/guide/${sectionId}`);
  };

  if (!content) {
    return (
      <div className="p-8">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-2xl font-bold text-slate-900 mb-4">Section Not Found</h1>
          <p className="text-slate-600">The requested section could not be found.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-slate-900 mb-4 leading-tight">
          {content.title}
        </h1>
        {currentSectionData && (
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              {currentSectionData.completed ? (
                <CheckCircle className="w-5 h-5 text-green-500" />
              ) : (
                <Circle className="w-5 h-5 text-slate-400" />
              )}
              <span className="text-sm text-slate-600">
                {currentSectionData.completed ? 'Completed' : 'In Progress'}
              </span>
            </div>
            {!currentSectionData.completed && (
              <button
                onClick={handleComplete}
                className="text-sm bg-green-100 text-green-700 px-3 py-1 rounded-full hover:bg-green-200 transition-colors"
              >
                Mark as Complete
              </button>
            )}
          </div>
        )}
      </div>

      {/* Content */}
      <div className="prose prose-slate max-w-none">
        <ReactMarkdown
          components={{
            code: ({ node, inline, className, children, ...props }) => {
              const match = /language-(\w+)/.exec(className || '');
              const language = match ? match[1] : '';
              const codeId = `code-${Math.random().toString(36).substr(2, 9)}`;
              
              if (!inline && match) {
                return (
                  <div className="code-block my-6">
                    <div className="code-header flex items-center justify-between">
                      <span className="text-slate-300">{language}</span>
                      <button
                        onClick={() => copyToClipboard(String(children).replace(/\n$/, ''), codeId)}
                        className="flex items-center space-x-1 text-slate-400 hover:text-slate-200 transition-colors"
                      >
                        {copiedCode === codeId ? (
                          <>
                            <Check className="w-4 h-4" />
                            <span className="text-xs">Copied</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-4 h-4" />
                            <span className="text-xs">Copy</span>
                          </>
                        )}
                      </button>
                    </div>
                    <SyntaxHighlighter
                      style={vscDarkPlus}
                      language={language}
                      PreTag="div"
                      className="code-content"
                      {...props}
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  </div>
                );
              }
              
              return (
                <code className="bg-slate-100 text-slate-800 px-1.5 py-0.5 rounded font-mono text-sm" {...props}>
                  {children}
                </code>
              );
            },
            h1: ({ children }) => (
              <h1 className="text-3xl font-bold text-slate-900 mt-8 mb-4 first:mt-0">
                {children}
              </h1>
            ),
            h2: ({ children }) => (
              <h2 className="text-2xl font-semibold text-slate-900 mt-8 mb-4">
                {children}
              </h2>
            ),
            h3: ({ children }) => (
              <h3 className="text-xl font-semibold text-slate-900 mt-6 mb-3">
                {children}
              </h3>
            ),
            p: ({ children }) => (
              <p className="text-slate-700 leading-relaxed mb-4">
                {children}
              </p>
            ),
            ul: ({ children }) => (
              <ul className="list-disc list-inside text-slate-700 mb-4 space-y-1">
                {children}
              </ul>
            ),
            ol: ({ children }) => (
              <ol className="list-decimal list-inside text-slate-700 mb-4 space-y-1">
                {children}
              </ol>
            ),
            li: ({ children }) => (
              <li className="text-slate-700">{children}</li>
            ),
            strong: ({ children }) => (
              <strong className="font-semibold text-slate-900">{children}</strong>
            ),
            table: ({ children }) => (
              <div className="overflow-x-auto my-6">
                <table className="min-w-full border border-slate-300 rounded-lg">
                  {children}
                </table>
              </div>
            ),
            th: ({ children }) => (
              <th className="bg-slate-100 border border-slate-300 px-4 py-2 text-left font-semibold text-slate-900">
                {children}
              </th>
            ),
            td: ({ children }) => (
              <td className="border border-slate-300 px-4 py-2 text-slate-700">
                {children}
              </td>
            ),
            blockquote: ({ children }) => (
              <blockquote className="border-l-4 border-blue-500 pl-4 italic text-slate-600 my-4">
                {children}
              </blockquote>
            ),
            a: ({ href, children }) => (
              <a
                href={href}
                className="text-blue-600 hover:text-blue-800 underline inline-flex items-center"
                target="_blank"
                rel="noopener noreferrer"
              >
                {children}
                <ExternalLink className="w-3 h-3 ml-1" />
              </a>
            )
          }}
        >
          {content.content}
        </ReactMarkdown>
      </div>

      {/* Navigation */}
      <div className="flex items-center justify-between mt-12 pt-8 border-t border-slate-200">
        <div>
          {previousSection && (
            <button
              onClick={() => goToSection(previousSection.id)}
              className="flex items-center space-x-2 text-slate-600 hover:text-slate-900 transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <div className="text-left">
                <div className="text-sm text-slate-500">Previous</div>
                <div className="font-medium">{previousSection.title}</div>
              </div>
            </button>
          )}
        </div>
        
        <div>
          {nextSection && (
            <button
              onClick={() => goToSection(nextSection.id)}
              className="flex items-center space-x-2 text-slate-600 hover:text-slate-900 transition-colors"
            >
              <div className="text-right">
                <div className="text-sm text-slate-500">Next</div>
                <div className="font-medium">{nextSection.title}</div>
              </div>
              <ArrowRight className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
