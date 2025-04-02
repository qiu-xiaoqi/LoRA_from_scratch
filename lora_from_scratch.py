import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

from tqdm import tqdm
from typing import List
from einops import rearrange
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM

device = 'cuda' if torch.cuda.is_available() else 'cpu' 
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
print(f'device:{device}\ndtype:{dtype}')

class LoraLinear(nn.Module):
    def __init__(
            self,
            base_layer: nn.Linear, # 原来的线性层
            r: int = 8, # lora Rank
            alpha: int = 16, # lora alpha
            dropout_p: float = 0.0, # lora dropout
            test_mode: bool = False, # 测试模式，用于控制lora_B 是否为全零
    ):
        super(LoraLinear, self).__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout_p)

        # 定义lora_A 和 lora_B 为 parameter
        self.lora_A = nn.Parameter(torch.empty((r, base_layer.in_features), dtype=base_layer.weight.dtype))
        self.lora_B = nn.Parameter(torch.empty((base_layer.out_features, r), dtype=base_layer.weight.dtype))

        # 初始化lora矩阵
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        if test_mode:
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)
        else:
            nn.init.zeros_(self.lora_B)

        for parma in self.base_layer.parameters():
            parma.requires_grad = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaling = float(self.alpha) / float(self.r) # lora缩放系数
        lora_adjustment = F.linear(self.dropout(x), self.lora_A)
        lora_adjustment = F.linear(lora_adjustment, self.lora_B)
        # 相当于 base_layer(x) + lora_A @ lora_B
        return self.base_layer(x) + lora_adjustment * scaling
    

def replace_linear_with_lora(
        module: nn.Module,
        r: int = 8,
        alpha: int = 16,
        dropout_p: float = 0.0,
        embed_requires_grad: bool = False, # embedding 层是否训练
        norm_requires_grad: bool = False, # norm 层是否训练
        head_requires_grad: bool = False, # head 层是否训练(Causal LM 才有)
        test_mode: bool = False, # 是否测试模式,用于控制lora_B 是否全为0
):
    """
    找到module中所有线性层并递归替换
    """
    for name, child in module.named_children():
        # 先处理额外的层, lm_head也是linear, 所以先处理
        if any(s in name for s in ['embed', 'norm', 'lm_head']):
            requires_grad = embed_requires_grad if 'embed' in name \
                            else norm_requires_grad if 'norm' in name \
                            else head_requires_grad
            for param in child.parameters():
                param.requires_grad = requires_grad

        # 替换所有线性层, QLoRA做法
        elif isinstance(child, nn.Linear):
            lora_linear = LoraLinear(child, r=r, alpha=alpha, dropout_p=dropout_p, test_mode=test_mode)
            setattr(module, name, lora_linear)
        # 递归向下替换
        else:
            replace_linear_with_lora(
                child,r,alpha,dropout_p,
                embed_requires_grad,norm_requires_grad,head_requires_grad,test_mode=test_mode
            )

def unload_lora(module: nn.Module, adapter_name: str='adapter'):
    """
    卸载lora参数, 并将原模型恢复至加载lora前的样子
    """
    lora_parameters = {}
    def search_lora_linear(module: nn.Module, pefix:List[str]):
        for name, child in module.named_children():
            new_prefix = pefix + [name]
            if isinstance(child, LoraLinear):
                # 保存lora参数
                lora_parameters['.'.join(new_prefix)] = {
                    "lora_A_weight": child.lora_A.data.cpu(),
                    "lora_B_weight": child.lora_B.data.cpu(),
                    "r": child.r,
                    "alpha": child.alpha,
                    "dropout_p": child.dropout.p
                }
                setattr(module, name, child.base_layer)
            else:
                search_lora_linear(child, new_prefix)
        
    search_lora_linear(module, [])
    # 解冻原模型
    for name, param in module.named_parameters():
        param.requires_grad = True
    torch.save(lora_parameters, f"{adapter_name}.pt")

def load_lora(module: nn.Module, adapter_name: str = 'adapter'):
    """
    加载lora参数
    """
    lora_parameters = torch.load(f"{adapter_name}.pt")

    for name, lora_params in lora_parameters.items():
        child = dict(module.named_modules())[name]
        if isinstance(child, nn.Linear):
            lora_linear = LoraLinear(child, lora_params['r'], lora_params['alpha'], lora_params['dropout_p'])
            lora_linear.lora_A.data = lora_params['lora_A_weight'].to(lora_linear.lora_A.device)
            lora_linear.lora_B.data = lora_params['lora_B_weight'].to(lora_linear.lora_B.device)

            # 名称示例： layers().self_attn.q_proj
            parts = name.split('.')
            obj = module
            for part in parts[:-1]: # 不包括最后一级
                obj = getattr(obj, part)
            setattr(obj, parts[-1], lora_linear)
    
    # 恢复原来的冻结方式，这里简单地除了lora全冻结
    for name, param in module.named_parameters():
        if any(s in name for s in ['embed', 'norm', 'lm_head']):
            param.requires_grad = False

def print_trainable_parameters(model: nn.Module):
    """
    打印可训练参数，和PeftModel的方法类似
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = 100 * trainable_params / total_params

    print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {trainable_percentage:.4f}")


if __name__ == '__main__':
    config = AutoConfig.for_model('llama') # 核心配置
    config.hidden_size = 24
    config.intermediate_size = config.hidden_size * 4
    config.num_attention_heads = 4
    config.num_hidden_layers = 4
    config.num_key_value_heads = 2
    config.vocab_size = 128
    bs = 2
    seq_len = 8
    test_tensor = torch.randint(0, config.vocab_size, (bs, seq_len))
    raw_model = AutoModel.from_config(config)
    
    lora_model = copy.deepcopy(raw_model)
    replace_linear_with_lora(lora_model, test_mode=True)
    
    raw_model.eval()
    print_trainable_parameters(raw_model)
    raw_res = raw_model(test_tensor).last_hidden_state
    
    # 第一次直接初始化 lora 的前向结果
    lora_model.eval()
    print_trainable_parameters(lora_model)  # 检查参数和可训练情况
    before_unload_res = lora_model(test_tensor).last_hidden_state
    
    # 卸载 lora 后的前向结果
    unload_lora(lora_model)
    lora_model.eval()
    print_trainable_parameters(lora_model)  # 检查参数和可训练情况
    unload_res = lora_model(test_tensor).last_hidden_state
    
    # 重新装载 lora 后的前向结果
    load_lora(lora_model)
    lora_model.eval()
    print_trainable_parameters(lora_model)  # 检查参数和可训练情况
    load_res = lora_model(test_tensor).last_hidden_state

    # 结果
    print(torch.allclose(raw_res, unload_res, atol=1e-6))           # 应为 True
    print(torch.allclose(before_unload_res, load_res, atol=1e-6))   # 应为 True
    print(torch.allclose(raw_res, load_res, atol=1e-6))  # 应为False