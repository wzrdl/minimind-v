"""
MiniMind-VLM (Vision-Language Model) 模型实现

本模块实现了基于 MiniMind 语言模型的视觉语言模型（VLM），
通过集成 CLIP 视觉编码器，使模型能够同时处理图像和文本输入。
主要功能包括：
- 图像编码和特征提取
- 视觉特征到语言模型空间的投影
- 多模态输入的前向传播
"""

import os

import torch
import warnings
from .model_minimind import *
from typing import Optional, Tuple, List
from torch import nn
from transformers import CLIPProcessor, CLIPModel
from typing import List

# 忽略警告信息，保持输出清洁
warnings.filterwarnings('ignore')


class VLMConfig(MiniMindConfig):
    """
    视觉语言模型配置类
    
    继承自 MiniMindConfig，用于配置 MiniMind-VLM 模型的参数。
    主要添加了图像相关的特殊 token 和 ID 配置。
    
    Attributes:
        model_type (str): 模型类型标识，固定为 "minimind-v"
        image_special_token (str): 用于表示图像的特殊 token 字符串，默认为 196 个 '@' 字符
        image_ids (List[int]): 图像在 token 序列中对应的 token ID 列表，默认为 196 个 ID 34
    """
    model_type = "minimind-v"

    def __init__(
            self,
            image_special_token: str = '@' * 196,
            image_ids: List = [34] * 196,
            **kwargs,
    ):
        """
        初始化 VLM 配置
        
        Args:
            image_special_token (str): 图像特殊 token，用于在文本序列中标记图像位置
                                      默认使用 196 个 '@' 字符（对应 CLIP ViT-Base 的 patch 数量）
            image_ids (List[int]): 图像 token 的 ID 列表，用于在 token 序列中识别图像位置
                                 默认使用 196 个 ID 34（对应 196 个图像 patch）
            **kwargs: 传递给父类 MiniMindConfig 的其他参数
        """
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        super().__init__(**kwargs)


class VisionProj(nn.Module):
    """
    视觉特征投影层
    
    将视觉编码器输出的特征向量投影到语言模型的隐藏空间维度。
    使用线性层将 CLIP 视觉编码器的输出（768 维）映射到语言模型的隐藏维度（512 维）。
    """
    
    def __init__(self, ve_hidden_size=768, hidden_size=512):
        """
        初始化视觉投影层
        
        Args:
            ve_hidden_size (int): 视觉编码器的隐藏层维度，CLIP ViT-Base 默认为 768
            hidden_size (int): 语言模型的隐藏层维度，默认为 512
        """
        super().__init__()
        self.ve_hidden_size = ve_hidden_size  # 视觉编码器隐藏维度
        self.hidden_size = hidden_size  # 目标隐藏维度（语言模型维度）
        # 使用线性层进行维度投影
        self.vision_proj = nn.Sequential(
            nn.Linear(self.ve_hidden_size, self.hidden_size)
        )

    def forward(self, image_encoders):
        """
        前向传播：将视觉编码特征投影到语言模型空间
        
        Args:
            image_encoders (torch.Tensor): 视觉编码器输出的特征向量
                                         形状: (batch_size, num_patches, ve_hidden_size)
        
        Returns:
            torch.Tensor: 投影后的特征向量，形状: (batch_size, num_patches, hidden_size)
        """
        vision_proj = self.vision_proj(image_encoders)
        return vision_proj


class MiniMindVLM(MiniMindForCausalLM):
    """
    MiniMind 视觉语言模型主类
    
    继承自 MiniMindForCausalLM，在语言模型基础上集成视觉编码能力。
    支持同时处理图像和文本输入，实现多模态理解和生成。
    
    主要组件：
    - vision_encoder: CLIP 视觉编码器，用于提取图像特征
    - processor: CLIP 图像处理器，用于预处理输入图像
    - vision_proj: 视觉特征投影层，将图像特征映射到语言模型空间
    """
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None, vision_model_path="./model/vision_model/clip-vit-base-patch16"):
        """
        初始化 MiniMind-VLM 模型
        
        Args:
            params (VLMConfig, optional): 模型配置对象，如果为 None 则使用默认配置
            vision_model_path (str): CLIP 视觉模型的路径，默认为 "./model/vision_model/clip-vit-base-patch16"
        """
        super().__init__(params)
        # 如果没有提供配置，使用默认配置
        if not params: 
            params = VLMConfig()
        self.params = params
        # 加载视觉编码器和图像处理器
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        # 初始化视觉投影层，将视觉特征投影到语言模型的隐藏维度
        self.vision_proj = VisionProj(hidden_size=params.hidden_size)

    @staticmethod
    def get_vision_model(model_path: str):
        """
        静态方法：加载 CLIP 视觉编码模型和处理器
        
        从指定路径加载预训练的 CLIP 模型，并冻结所有参数（不进行梯度更新）。
        视觉编码器在训练过程中保持固定，只训练投影层和语言模型部分。
        
        Args:
            model_path (str): CLIP 模型文件路径
        
        Returns:
            tuple: (vision_model, processor) 元组
                   - vision_model: CLIP 视觉编码器模型（评估模式，参数冻结）
                   - processor: CLIP 图像处理器
                   如果模型路径不存在，返回 (None, None)
        """
        from transformers import logging as hf_logging
        # 设置 transformers 日志级别为错误，减少输出信息
        hf_logging.set_verbosity_error()
        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            return None, None
        # 加载预训练的 CLIP 模型和处理器
        model = CLIPModel.from_pretrained(model_path)
        processor = CLIPProcessor.from_pretrained(model_path)
        # 冻结 vision_encoder 的所有参数，在训练过程中不更新
        for param in model.parameters():
            param.requires_grad = False
        # 设置为评估模式并返回
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        """
        静态方法：将 PIL 图像转换为模型输入张量
        
        将图像转换为 CLIP 模型可以处理的张量格式。
        如果图像是 RGBA 或 LA 模式，会先转换为 RGB 模式。
        
        Args:
            image: PIL Image 对象，输入的图像
            processor: CLIPProcessor 对象，用于处理图像
        
        Returns:
            torch.Tensor: 处理后的图像张量，形状: (1, 3, H, W)
        """
        # 如果图像是 RGBA（带透明度）或 LA（灰度+透明度）模式，转换为 RGB
        if image.mode in ['RGBA', 'LA']: 
            image = image.convert('RGB')
        # 使用 CLIP 处理器将图像转换为张量
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        """
        静态方法：从图像张量中提取视觉特征嵌入
        
        使用 CLIP 视觉编码器提取图像的视觉特征。
        跳过 CLS token（第一个 token），只返回图像 patch 的特征。
        
        Args:
            image_tensors (torch.Tensor): 预处理后的图像张量
                                        形状: (batch_size, 3, H, W)
            vision_model: CLIP 视觉编码器模型
        
        Returns:
            torch.Tensor: 图像特征嵌入，形状: (batch_size, num_patches, hidden_size)
                         对于 ViT-Base，num_patches = 196（14x14）
        """
        # 使用 no_grad 上下文，因为视觉编码器参数已冻结，不需要计算梯度
        with torch.no_grad():
            # 通过视觉编码器获取图像特征
            outputs = vision_model.vision_model(pixel_values=image_tensors)
        # 跳过 CLS token（索引 0），只取图像 patch 的特征（索引 1:）
        # squeeze() 用于移除可能的单维度
        img_embedding = outputs.last_hidden_state[:, 1:, :].squeeze()
        return img_embedding

    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        """
        将视觉特征投影并插入到隐藏状态中
        
        在 token 序列中找到图像 token 的位置，并用投影后的视觉特征替换对应的隐藏状态。
        这是实现多模态融合的关键步骤：将图像特征嵌入到文本序列的相应位置。
        
        Args:
            tokens (torch.Tensor): 输入 token ID 序列，形状: (batch_size, seq_length)
            h (torch.Tensor): 文本 token 的隐藏状态，形状: (batch_size, seq_length, hidden_size)
            vision_tensors (torch.Tensor, optional): 视觉编码器输出的特征，形状: (batch_size, num_patches, ve_hidden_size)
            seqlen (int): 序列长度限制，默认为 512
        
        Returns:
            torch.Tensor: 融合了视觉特征的隐藏状态，形状: (batch_size, seq_length, hidden_size)
                         如果 vision_tensors 为 None 或未找到图像 token，返回原始隐藏状态 h
        """
        def find_indices(tokens, image_ids):
            """
            内部函数：在 token 序列中查找图像 token 的位置
            
            使用滑动窗口方法在 token 序列中查找连续的图像 token ID 序列。
            返回每个批次中所有图像 token 的起始和结束位置。
            
            Args:
                tokens (torch.Tensor): token ID 序列，形状: (batch_size, seq_length)
                image_ids (List[int]): 图像 token ID 列表（如 [34, 34, ..., 34]）
            
            Returns:
                dict or None: 字典，键为批次索引，值为 (start_idx, end_idx) 元组列表
                             如果未找到匹配或序列太短，返回 None
            """
            # 将图像 ID 列表转换为张量，并移动到与 tokens 相同的设备
            image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
            len_image_ids = len(image_ids)
            # 如果图像 token 序列长度大于 token 序列长度，无法匹配
            if len_image_ids > tokens.size(1):
                return None
            # 使用 unfold 创建滑动窗口视图，在每个位置检查是否匹配图像 token 序列
            tokens_view = tokens.unfold(1, len_image_ids, 1)
            # 检查每个窗口是否完全匹配图像 token ID 序列
            matches = (tokens_view == image_ids_tensor).all(dim=2)
            # 构建结果字典：批次索引 -> [(start_idx, end_idx), ...]
            return {
                batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in
                            matches[batch_idx].nonzero(as_tuple=True)[0]]
                for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()
            } or None

        # 在 token 序列中查找图像 token 的位置
        image_indices = find_indices(tokens, self.params.image_ids)
        
        # 如果提供了视觉特征且找到了图像 token 位置，进行融合
        if vision_tensors is not None and image_indices:
            # 将视觉特征投影到语言模型的隐藏空间
            vision_proj = self.vision_proj(vision_tensors)
            # 如果维度为 3，添加批次维度（处理单批次情况）
            if len(vision_proj.shape) == 3:
                vision_proj = vision_proj.unsqueeze(0)
            
            new_h = []  # 存储融合后的隐藏状态
            # 遍历每个批次
            for i in range(h.size(0)):
                if i in image_indices:
                    # 如果该批次包含图像 token，进行替换
                    h_i = h[i]  # 当前批次的隐藏状态
                    img_idx = 0  # 图像特征索引
                    # 遍历该批次中所有图像 token 的位置
                    for start_idx, end_idx in image_indices[i]:
                        if img_idx < vision_proj.size(1):
                            # 将图像 token 位置的隐藏状态替换为投影后的视觉特征
                            # 拼接：[:start_idx] + 视觉特征 + [end_idx+1:]
                            h_i = torch.cat((h_i[:start_idx], vision_proj[i][img_idx], h_i[end_idx + 1:]), dim=0)[
                                  :seqlen]  # 截断到指定序列长度
                            img_idx += 1
                    new_h.append(h_i)
                else:
                    # 如果该批次不包含图像 token，保持原样
                    new_h.append(h[i])
            # 堆叠所有批次的隐藏状态
            return torch.stack(new_h, dim=0)
        # 如果没有视觉特征或未找到图像 token，返回原始隐藏状态
        return h

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        """
        前向传播：处理多模态输入（文本 + 图像）并生成输出
        
        这是模型的核心方法，实现了以下流程：
        1. Token 嵌入：将输入 token ID 转换为嵌入向量
        2. 视觉特征提取：如果提供了图像，提取并投影视觉特征
        3. 多模态融合：将视觉特征插入到文本序列的相应位置
        4. 位置编码：添加位置信息
        5. 层间传播：通过 Transformer 层处理
        6. 输出生成：生成 logits 和辅助损失
        
        Args:
            input_ids (torch.Tensor, optional): 输入 token ID 序列，形状: (batch_size, seq_length)
            attention_mask (torch.Tensor, optional): 注意力掩码，用于屏蔽填充 token
            past_key_values (List[Tuple], optional): 缓存的键值对，用于加速生成
            use_cache (bool): 是否使用缓存，默认为 False
            logits_to_keep (Union[int, torch.Tensor]): 保留的 logits 数量或索引，默认为 0（只保留最后一个）
            pixel_values (torch.FloatTensor, optional): 图像像素值，形状: (batch_size, num_images, 3, H, W)
            **args: 其他可选参数
        
        Returns:
            dict: 包含以下键的输出字典：
                - 'last_hidden_state': 最后一层的隐藏状态
                - 'logits': 语言模型输出的 logits
                - 'aux_loss': 辅助损失（如果使用 MoE 层）
                - 'past_key_values': 缓存的键值对（如果 use_cache=True）
        """
        # 获取批次大小和序列长度
        batch_size, seq_length = input_ids.shape
        
        # 处理 past_key_values：如果是特殊对象类型，重置为 None
        if hasattr(past_key_values, 'layers'): 
            past_key_values = None
        # 如果 past_key_values 为 None，初始化为每层都是 None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        # 计算起始位置：如果有缓存的键值对，从缓存长度开始；否则从 0 开始
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 步骤 1: Token 嵌入 - 将 token ID 转换为嵌入向量并应用 dropout
        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        # 步骤 2 & 3: 视觉特征提取和多模态融合
        # 只在首次前向传播时处理图像（start_pos == 0），避免重复处理
        if pixel_values is not None and start_pos == 0:
            # 如果像素值有 6 个维度，压缩掉多余的维度
            if len(pixel_values.shape) == 6:
                pixel_values = pixel_values.squeeze(2)
            # 获取形状信息：批次大小、图像数量、通道数、高度、宽度
            bs, num, c, im_h, im_w = pixel_values.shape
            # 根据批次大小决定堆叠维度
            stack_dim = 1 if bs > 1 else 0
            # 对每张图像提取视觉特征并堆叠
            vision_tensors = torch.stack([
                MiniMindVLM.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)
                for i in range(num)
            ], dim=stack_dim)
            # 将视觉特征投影并融合到隐藏状态中
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors,
                                                   seqlen=input_ids.shape[1])

        # 步骤 4: 位置编码 - 获取当前序列位置对应的正弦和余弦位置编码
        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        # 步骤 5: 通过 Transformer 层进行前向传播
        presents = []  # 存储每层的键值对（用于缓存）
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            # 通过每一层处理隐藏状态
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # 步骤 6: 层归一化 - 对最后一层的隐藏状态进行归一化
        hidden_states = self.model.norm(hidden_states)

        # 计算辅助损失：如果使用 MoE（Mixture of Experts）层，累加所有层的辅助损失
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.model.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )
        
        # 根据 logits_to_keep 参数决定保留哪些位置的 logits
        # 如果是整数，保留最后 N 个；如果是张量，使用指定的索引
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # 通过语言模型头生成 logits
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        # 将结果存储到输出字典中
        self.OUT.__setitem__('last_hidden_state', hidden_states)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', presents)
        return self.OUT
