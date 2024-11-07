import scanpy as sc
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import torch
import pynvml
import operator
import anndata
import warnings
import random
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.neighbors import NearestNeighbors
from tokenfilter import TokenFilter

class glitchhunter:
    
    def __init__(self, 
                 model_name:str, 
                 auth_token = 'None'
                ):  
        self.model_name = model_name
        self.auth_token = auth_token
    
    
                
    def load_model(self,device='cuda'):
        
        if self.auth_token == 'None':
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,device_map=device, torch_dtype=torch.bfloat16,)    
            except:
                self.model = AutoModel.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code = True,device_map=device, torch_dtype=torch.bfloat16,)
        else:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, use_auth_token = self.auth_token)
            except:
                self.model = AutoModel.from_pretrained(self.model_name, use_auth_token = self.auth_token)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token = self.auth_token)
            
        global wte
        try:
            wte = self.model.transformer.word_embeddings
        except:
            try:
                wte = self.model.wte
            except:
                try:
                    wte = self.model.model.embed_tokens
                except:
                    try:
                        wte = self.model.transformer.embedding.word_embeddings
                    except:
                        None
        
        if 'chatglm' in self.model_name:
            self.model = self.model.half()
        self.wte = wte.weight.detach().float().cpu().numpy()
        self.vocab_size = self.tokenizer.vocab_size
        self.ad = sc.AnnData(self.wte)[0:self.vocab_size]
        self.model.eval()
                            
    
    def __generate_response(self, input_text:str, tokens):
        if 'chatglm' in self.model_name:
            self.model = self.model.eval()
            response, history = self.model.chat(self.tokenizer, input_text)
            return response
            
        else:
            output = self.model.generate(
                tokens,
                max_new_tokens=10,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id = self.tokenizer.eos_token_id
            )[0][list(tokens[0].shape)[0]:]
            response = self.tokenizer.decode(output, skip_special_tokens=True)
            return response
    
    def __repetition_test(self, token_id, token:str):
        string_to_repeat = token

        
        if 'Mistral' in self.model_name:
            quote_id = 28742
        elif 'chatglm2' in self.model_name:
            quote_id = 30953
        elif 'chatglm' in self.model_name:
            quote_id = 22
        else:
            quote_id = 29915
        text1 = "Question: Please can you repeat the string '"
        text2 = "' and return back to me?\n\nAnswer: Of course! Here is the repeated string:"
        tokens1 = torch.tensor(self.tokenizer.encode(text1))
        tokens2 = torch.tensor(self.tokenizer.encode(text2))
        tokens = torch.cat((tokens1, torch.tensor([token_id]), torch.tensor([quote_id]), tokens2[2:]), dim=0).to(self.model.device)
        text = f"Question: Please can you repeat the string '{token}' and return back to me?\n\nAnswer: Of course! Here is the repeated string:"
        tokens = torch.unsqueeze(tokens, dim=0)
        
        response = self.__generate_response(text, tokens)
        if string_to_repeat in response:
            return (response, True)
        string_to_repeat = string_to_repeat.upper()
        if string_to_repeat in response.upper():
            return (response, True)
        #print(string_to_repeat + ' falied in ' + response)
        return (response, False)

    
    def identify_glitch_token(self, threshold=0, k=50, gamma=75, max_tokens=None):
        n = self.vocab_size
        self.ad = self.ad.copy()
        self.ad.obs["index"] = range(n)
        ad_b = self.ad.copy()
        
        self.glitch_tokens = set()
        self.non_glitch_tokens = set()
        judge_num = 0
        all_tokens = set()  # 用于存储所有检测到的 token
        token_count = 0  # 用于计数检测到的 token 数量
        
        token_filter = TokenFilter(self.model, self.tokenizer)
        skip_tokens = token_filter.filter_token()
        
        while True:
            # PCA
            n_components = min(50, ad_b.X.shape[1])
            ipca = IncrementalPCA(n_components=n_components, batch_size=200)
            ad_b.obsm['X_pca'] = ipca.fit_transform(ad_b.X)
            
            sc.pp.neighbors(ad_b, n_neighbors=k, use_rep='X_pca')
            sc.tl.louvain(ad_b, resolution=gamma)
            
            ad_glitch_can = []
            cluster_num = ad_b.obs['louvain'].nunique()
            split = [ad_b[ad_b.obs['louvain'] == str(i)] for i in range(cluster_num)]
            
            for cluster in range(cluster_num):
                ad_sub = split[cluster]
                tokens_i = ad_sub.shape[0]
                ad_sub_ = sc.pp.subsample(ad_sub, n_obs=min(max(5, int(min(0.1 * tokens_i, 50))), tokens_i), copy=True, random_state=random.randint(0, 199))
                
                cal = 0
                k_list = set(map(int, ad_sub_.obs.index))
                cal += len(k_list & self.glitch_tokens)
                k_list -= self.glitch_tokens
                k_list -= self.non_glitch_tokens
                
                if cal / ad_sub_.shape[0] > threshold:
                    ad_glitch_can.append(ad_sub)
                    continue
                
                for token_id in k_list:
                    if token_id not in skip_tokens:
                        token = self.tokenizer.decode([token_id])
                        response, sem = self.__repetition_test(token_id, token)
                        if not sem:
                            self.glitch_tokens.add(token_id)
                            cal += 1
                            is_glitch = True
                        else:
                            self.non_glitch_tokens.add(token_id)
                            is_glitch = False
                        judge_num += 1
                        all_tokens.add(token_id)  # 添加到所有 token 集合中
                        token_count += 1  # 增加 token 计数

                        # 打印检测到的 token 信息
                        print(f"Token {token_count}: ID = {token_id}, Is Glitch = {is_glitch}")

                        # Check if we have reached the max_tokens limit
                        if max_tokens and len(all_tokens) >= max_tokens:
                            glitch_probability = len(self.glitch_tokens) / token_count
                            print(f"Glitch token probability: {glitch_probability:.2%}")
                            return list(all_tokens)
                
                if cal / ad_sub_.shape[0] > threshold:
                    ad_glitch_can.append(ad_sub)
                        
            ad_glitch_can = anndata.concat(ad_glitch_can)
            if ad_glitch_can.shape[0] == ad_b.shape[0]:
                break
            else:
                ad_b = ad_glitch_can.copy()
        
        k_list = set(map(int, ad_b.obs.index)) - self.non_glitch_tokens
        ad_b = ad_b[[i for i in range(len(k_list)) if str(k_list[i]) in ad_b.obs.index]]
        self.final_giltch_token = ad_b
        print(judge_num)
        
        # 计算并打印故障 token 的概率
        glitch_probability = len(self.glitch_tokens) / token_count
        print(f"Glitch token probability: {glitch_probability:.2%}")
        
        return list(all_tokens)
                    
    def testing_glitch_tokens(self):
        
        cal = 0
        k_list = list(map(int, list(self.final_giltch_token.obs.index)))
        cal += len(set(k_list) & set(self.glitch_tokens))
        k_list = list(set(k_list) - set(self.glitch_tokens))
        for token_id in k_list:
            token = self.tokenizer.decode([token_id])
            bonus_response, sem = self.__repetition_test_bonus(token)
            if (sem == False):
                response, sem = self.__repetition_test(token)
                if (sem == False):
                    self.glitch_tokens.append(token_id)
                    judge = True
                else:
                    self.non_glitch_tokens.append(token_id)
                    judge = False
            else:
                self.non_glitch_tokens.append(token_id)
                judge = False

            if judge:
                cal += 1
        return cal, len(list(self.final_giltch_token.obs.index)), cal/len(list(self.final_giltch_token.obs.index))