import torch
import torch.nn as nn

# 用于调试维度问题的简单测试
def debug_dimensions():
    # 创建测试张量
    batch_size = 2
    seq_length = 13  # 13个模态
    hidden_dim = 256
    
    # 模拟combined张量
    combined = torch.randn(batch_size, seq_length, hidden_dim)
    print(f"Combined shape: {combined.shape}")
    
    # 创建位置编码嵌入层
    position_embeddings = nn.Embedding(100, hidden_dim)
    
    # 生成位置ID
    position_ids = torch.arange(seq_length, dtype=torch.long)
    print(f"Position IDs shape: {position_ids.shape}")
    
    # 获取位置嵌入
    pos_embed = position_embeddings(position_ids)
    print(f"Position embeddings shape: {pos_embed.shape}")
    
    # 扩展维度以匹配batch_size
    pos_embed = pos_embed.unsqueeze(0).expand(batch_size, -1, -1)
    print(f"Expanded position embeddings shape: {pos_embed.shape}")
    
    # 测试相加
    result = combined + pos_embed
    print(f"Addition result shape: {result.shape}")
    
    print("All operations successful!")

# 运行调试
debug_dimensions()
