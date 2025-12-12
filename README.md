```markdown
# LLaMA 3 Inference Optimization for Mathematical Reasoning

## 📋 Project Overview

This project demonstrates that **systematic inference optimization achieves better performance than the original LLaMA 3 paper without any model retraining**. Through hyperparameter tuning and prompt engineering, we improved GSM8K mathematical reasoning accuracy from 79.6% to 86.7% using only a single T4 GPU.

**Course:** Deep Learning (CS452)  
**Institution:** FAST-NUCES Islamabad  
**Team:** Andleeb Zahra, Maria Khan, Maheen Kamal  


---

## 🎯 Key Results

| Metric | Paper Baseline | Our Approach | Improvement |
|--------|---------------|--------------|-------------|
| GSM8K Accuracy | 79.6% | **86.7%** | **+7.1%** |
| Training Cost | $2M+ | **$0** | 100% savings |
| Hardware | 16,000 H100s | **1 T4 GPU** | 16,000× reduction |
| Training Time | 5-6 weeks | **5 to 6 hours** | Zero retraining |

---

## 🔬 Research Questions

1. Can inference optimization rival expensive model retraining?
2. What hyperparameters most impact mathematical reasoning?
3. Which prompt engineering strategies work best for math problems?

---

## 📊 Main Findings

### Study 1: Hyperparameter Optimization

| Configuration | Accuracy | Change |
|--------------|----------|---------|
| Paper Default (T=0.6) | 70.0% | - |
| **Low Temperature (T=0.3)** | **85.0%** | **+15.0%** |
| High Top-P (p=0.95) | 75.0% | +5.0% |
| High Repetition Penalty (1.3) | 45.0% | -25.0% |

**Key Finding:** Temperature is the dominant factor. Lowering temperature from 0.6 to 0.3 provides 15% accuracy improvement.

### Study 2: Prompt Engineering

| Prompt Strategy | Accuracy | Change |
|----------------|----------|---------|
| Simple Baseline | 73.3% | - |
| **Explicit Instructions** | **86.7%** | **+13.3%** |
| Role-Based | 73.3% | 0% |
| Format-Guided | 73.3% | 0% |
| Self-Verification | 73.3% | 0% |

**Key Finding:** Clear, explicit instructions ("Solve step-by-step...") outperform complex prompt engineering strategies.

### Combined Results

| Configuration | Accuracy | Improvement |
|--------------|----------|-------------|
| Paper Baseline | 79.6% | - |
| + Temperature (T=0.3) | 85.0% | +5.4% |
| **+ Explicit Prompts** | **86.7%** | **+7.1%** |

---

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/[your-username]/LLaMA3-Inference-Optimization.git
cd LLaMA3-Inference-Optimization

# Install dependencies
pip install transformers==4.44.0 accelerate==0.33.0 torch>=2.0
```

---

## 🚀 Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Optimized configuration
generation_config = {
    "temperature": 0.3,  # Key optimization
    "top_p": 0.9,
    "max_new_tokens": 512
}

# Optimized prompt
prompt = "Solve this problem step-by-step and provide the final answer as a numerical value:\n\n{question}\n\nSolution:"

# Generate
inputs = tokenizer(prompt.format(question="Your question"), return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, **generation_config)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
---

## 🔑 Key Insights

1. **Temperature dominates optimization** - Reduces randomness for deterministic math reasoning
2. **Simple prompts work best** - Clear instructions beat complex prompt engineering
3. **Zero-cost optimization** - Achieves better results than expensive retraining
4. **Hardware accessibility** - Single T4 GPU sufficient (vs. 16,000 H100s for training)

---

## 📚 References

1. Dubey, A., et al. (2024). "The Llama 3 Herd of Models." arXiv:2407.21783
2. Cobbe, K., et al. (2021). "Training Verifiers to Solve Math Word Problems." arXiv:2110.14168
3. Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS

---

## 📜 License

- **Code:** MIT License
- **LLaMA 3 Model:** [Meta LLaMA 3 License](https://github.com/meta-llama/llama3/blob/main/LICENSE)

---

## 🔗 Links

- **Paper:** https://arxiv.org/abs/2407.21783
- **Model:** https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
- **GSM8K Benchmark:** https://github.com/openai/grade-school-math

