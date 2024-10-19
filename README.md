# 🧑‍🚀⛏️👻 GlitchMiner: Efficient Glitch Token Detection in Large Language Models

![Python](https://img.shields.io/badge/Python-3.x-blue.svg?style=flat-square)  
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange?style=flat-square)  
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen?style=flat-square)  

**GlitchMiner** is an innovative tool designed to detect **glitch tokens** in large language models (LLMs). These tokens can trigger unexpected model behaviors, such as incorrect outputs, hallucinations, or security vulnerabilities.

---

## 🧐 What Are Glitch Tokens?
<p align="center">
<img src="https://github.com/user-attachments/assets/dc7320ef-7e0f-4f5c-bca6-1e057532dc33" alt="llama2_glitch" width="50%">
</p>


**Glitch tokens** are anomalous tokens that a model fails to process correctly, potentially leading to:
- **Incorrect outputs** in high-stakes applications (e.g., healthcare or finance).
- **Hallucinated content**, reducing the model's reliability.
- **Security risks**, as attackers may exploit them for **Jailbreak** attacks.

### Possible Causes of Glitch Tokens:
- **Insufficient training** on rare tokens.
- **Poor embedding representation** in the model.
- **Low token frequency**, causing the model to ignore them.

---

## 🔍 GlitchMiner Overview
![glitchminer](https://github.com/user-attachments/assets/fd6d3a48-3b1f-452a-b77b-03b0748ab63e)

**GlitchMiner** is a cutting-edge tool designed to identify and mitigate **glitch tokens** in large language models (LLMs). These tokens, often caused by under-training or poor embedding alignment, can result in unpredictable behavior, such as incorrect outputs, hallucinations, or even exploitable vulnerabilities. By efficiently detecting these problematic tokens, GlitchMiner helps enhance the reliability and security of language models.

### How Does GlitchMiner Work?

GlitchMiner leverages **gradient-based discrete optimization** to systematically explore the token space. It focuses on maximizing **entropy**, targeting regions where the model is most uncertain about its predictions. Through this process, the tool identifies glitch tokens that may lead to erroneous outputs or security flaws.

Key steps include:
1. **Token Entropy Measurement:** GlitchMiner computes the entropy of token predictions to find areas of high uncertainty.
2. **Gradient-Based Search:** It uses gradients to guide the search towards glitch tokens, enhancing the efficiency of the detection process.
3. **Adaptive Filtering:** GlitchMiner refines the token space dynamically, ensuring only relevant tokens are evaluated.
4. **Multi-Model Support:** It seamlessly integrates with various LLMs, including Llama, Qwen, and Gemma, for flexible usage.

### Why Use GlitchMiner?

- **Security Assurance:** Detect and address glitch tokens that may expose vulnerabilities.
- **Performance Enhancement:** Improve the accuracy and reliability of your models by eliminating unpredictable behaviors.
- **Easy Integration:** GlitchMiner directly supports Hugging Face models.
- **Scalable and Efficient:** The tool is optimized for both small and large models, making it suitable for research and production environments.

With GlitchMiner, researchers and developers can confidently deploy LLMs in critical applications, knowing that potential glitches have been identified and mitigated.


---

## 🛠️ Quick Start

Install GlitchMiner via `pip`:

```bash
pip install git+https://github.com/wooozihui/GlitchMiner.git
```
### Example Usage

Here is a simple example of how to load a model and use GlitchMiner:

```python
from glitchminer import GlitchMiner, initialize_model_and_tokenizer
import time

if __name__ == "__main__":
    # Load a pre-trained model (Gemma-2-2b-it in this case)
    model_path = "google/gemma-2-2b-it"
    model, tokenizer = initialize_model_and_tokenizer(
        model_path, device="auto", quant_type="bfloat16"
    )

    start_time = time.time()

    # Run GlitchMiner for glitch token detection
    glitch_tokens, glitch_token_ids = GlitchMiner(
        model,
        tokenizer,
        num_iterations=125,
        batch_size=8,
        k=32,
        if_print=True,
        print_language="CN",
    )

    # Display runtime
    end_time = time.time()
    runtime = end_time - start_time
    print(f"GlitchMiner runtime: {runtime:.2f} seconds")
```
## ⚙️ GlitchMiner Parameters

Here are the configurable parameters for GlitchMiner, with explanations of their purpose and usage:

| Parameter         | Type     | Default Value | Description                                                                                       |
|-------------------|----------|---------------|---------------------------------------------------------------------------------------------------|
| `model`           | Model    | **Required**  | A Hugging Face `AutoModelForCausalLM` model used for glitch token detection.                       |
| `tokenizer`       | Tokenizer| **Required**  | A Hugging Face `AutoTokenizer` for encoding and decoding tokens.                                   |
| `num_iterations`  | `int`    | 125            | The number of iterations to run the glitch token search.                                           |
| `batch_size`      | `int`    | 8             | Number of tokens processed per batch during the search process.                                    |
| `k`               | `int`    | 32            | Number of top similar tokens to evaluate during each iteration using cosine similarity.            |
| `if_print`        | `bool`   | `True`        | If `True`, prints detailed progress and results during execution.                                  |
| `print_language`  | `str`    | `"CN"`        | Output language for printed messages. Supports `"CN"` for Chinese and `"ENG"` for English.          |
| `skip_tokens`     | `list`   | `None`        | Optional list of token IDs to exclude from the glitch detection process.                           |


## 📚 Key Features

- **Gradient-based discrete optimization:** Leverages model gradients to efficiently detect glitch tokens.
- **Entropy-driven strategy:** Focuses on areas with high prediction uncertainty.
- **Multi-model support:** Compatible with popular models like Llama, Qwen, Gemma, and more.
- **Customizable templates:** Allows you to tailor input formats for various use cases.

---

## 📊 Experimental Results

Below are some key results demonstrating GlitchMiner’s superior performance:

| Model             | Precision@1000 | Precision@2000 |
|-------------------|----------------|----------------|
| Llama-3.1-8B      | 38.2%          | 49.87%         |
| Qwen2.5-7B        | 45.47%         | 44.89%         |
| Gemma-2-9b-it     | 90.17%         | 70.57%         |
| Mistral-Nemo      | 57.03%         | 61.48%         |

For more details, refer to our paper (coming soon).


---

## 🧑‍💻 Authors

- **Zihui Wu** - Xidian University  
  - Email: [zihui@stu.xidian.edu.cn](mailto:zihui@stu.xidian.edu.cn)  
- **Haichang Gao** - Xidian University  
  - Email: [hchgao@xidian.edu.cn](mailto:hchgao@xidian.edu.cn)  
- **Ping Wang** - Xidian University  
- **Shudong Zhang** - Xidian University  
- **Zhaoxiang Liu** - China Unicom  
- **Shiguo Lian** - China Unicom  

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).






