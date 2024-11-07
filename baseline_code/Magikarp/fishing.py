import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_template import get_template_for_model
from tokenfilter import TokenFilter
import torch.nn.functional as F

def is_glitch_token(model, tokenizer, token_id, chat_template=None):
    # 获取模型设备
    device = model.device
    
    # 解码 token_id 获取对应的文本
    token = tokenizer.decode([token_id])
    
    # 如果未提供 chat_template，则根据模型名称获取
    if chat_template is None:
        chat_template = get_template_for_model(model.config._name_or_path)
    
    # 获取模板格式
    system_format = chat_template.system_format
    user_format = chat_template.user_format
    assistant_prefill = ' Sure, the string is: "«'
    system_message = ''
    
    # 构建用户提示
    user_prompt = f'Please repeat the string: "«{token}»"'
    
    # 格式化输入
    formatted_system = system_format.format(content=system_message) if system_format else ""
    formatted_user = user_format.format(content=user_prompt)
    formatted_input = formatted_system + formatted_user + assistant_prefill
    
    # 对输入进行分词
    input_ids = tokenizer.tokenizer.encode(formatted_input, return_tensors='pt').to(device)
    
    # 获取模型的输出
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        # 获取下一个 token 的 logits
        next_token_logits = outputs.logits[:, -1, :]
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        # 获取概率最大的 token 的 id
        predicted_token_id = next_token_probs.argmax(dim=-1).item()
        
        # 判断预测的 token 是否与输入的 token 一致
        is_glitch = predicted_token_id != token_id
        
    return is_glitch

# 定义生成候选 token 的函数
def find_glitch_candidates(model_name, k):
    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(model_name,device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    token_filter = TokenFilter(model, tokenizer)
    skip_tokens = token_filter.filter_token()
    # 获取模型配置，判断是否为tied embedding
    is_tied_embedding = model.config.tie_word_embeddings if hasattr(model.config, 'tie_word_embeddings') else False
    
    candidate_tokens = []
    embedding_matrix = model.get_input_embeddings().weight.detach().cpu()
    vocab_size = embedding_matrix.size(0)

    if is_tied_embedding:
        print("Model is tied embedding.")

        first_glitch_token = None
        while first_glitch_token is None:
            token_id = random.randint(0, vocab_size - 1)
            if token_id not in skip_tokens and is_glitch_token(model, token_filter, token_id):
                first_glitch_token = token_id

            candidate_tokens.append(token_id)
            if len(candidate_tokens) == k:
                return candidate_tokens

        unembedding_matrix = model.get_output_embeddings().weight.detach().cpu()
        first_token_embedding = unembedding_matrix[first_glitch_token]

        similarities = []
        for token_id in range(vocab_size):
            if token_id == first_glitch_token or token_id in skip_tokens:
                continue
            token_embedding = unembedding_matrix[token_id]
            cosine_sim = F.cosine_similarity(first_token_embedding, token_embedding, dim=0)
            similarities.append((token_id, cosine_sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        i = 0
        while len(candidate_tokens) < k and i < len(similarities):
            candidate_tokens.append(similarities[i][0])
            i += 1

    else:
        print("Model is not tied embedding.")

        norms = []
        for token_id in range(vocab_size):
            if token_id in skip_tokens:
                continue
            l2_norm = torch.norm(embedding_matrix[token_id], p=2)
            norms.append((token_id, l2_norm))

        norms.sort(key=lambda x: x[1])
        i = 0
        while len(candidate_tokens) < k and i < len(norms):
            candidate_tokens.append(norms[i][0])
            i += 1

    return candidate_tokens
    # 保存候选 token 的列表
    #torch.save(candidate_tokens, f"{model_name}_glitch_candidates.pt")
    #print(f"Saved {k} candidate tokens for {model_name}.")
