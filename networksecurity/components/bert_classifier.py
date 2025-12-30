"""
BERT文本分类模型
用于网页文本的钓鱼检测
"""

import os
import sys
import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%H:%M:%S')

# 检查transformers是否可用
try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from transformers import (
        DistilBertTokenizer, 
        DistilBertModel, 
        DistilBertForSequenceClassification,
        get_linear_schedule_with_warmup
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    logging.warning(f"transformers或torch未安装，BERT功能不可用: {e}")


class PhishingTextDataset(Dataset):
    """钓鱼文本数据集"""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, 
                 tokenizer=None, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item


class BertPhishingClassifier:
    """基于BERT的钓鱼网页文本分类器"""
    
    MODEL_NAME = 'distilbert-base-uncased'
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        初始化分类器
        
        Args:
            model_path: 预训练模型路径（如果有）
            device: 计算设备 ('cuda' 或 'cpu')
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers和torch未安装，请运行: pip install transformers torch")
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logging.info(f"使用设备: {self.device}")
        
        # 加载tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.MODEL_NAME)
        
        # 加载或创建模型
        if model_path and os.path.exists(model_path):
            logging.info(f"从 {model_path} 加载模型")
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        else:
            logging.info(f"创建新的 {self.MODEL_NAME} 模型")
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.MODEL_NAME,
                num_labels=2
            )
        
        self.model.to(self.device)
        self.max_length = 256
    
    def freeze_bert_layers(self, num_layers_to_freeze: int = 4):
        """
        冻结BERT的部分层，只微调最后几层
        
        Args:
            num_layers_to_freeze: 要冻结的transformer层数（DistilBERT有6层）
        """
        # 冻结embeddings
        for param in self.model.distilbert.embeddings.parameters():
            param.requires_grad = False
        
        # 冻结前N层transformer
        for i in range(num_layers_to_freeze):
            for param in self.model.distilbert.transformer.layer[i].parameters():
                param.requires_grad = False
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logging.info(f"冻结了 {num_layers_to_freeze} 层transformer")
        logging.info(f"可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: Optional[List[str]] = None, val_labels: Optional[List[int]] = None,
              epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5,
              freeze_layers: int = 4) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            train_texts: 训练文本列表
            train_labels: 训练标签列表 (0: 安全, 1: 钓鱼)
            val_texts: 验证文本列表
            val_labels: 验证标签列表
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            freeze_layers: 冻结的层数
            
        Returns:
            训练历史
        """
        logging.info("开始训练BERT钓鱼分类器...")
        
        # 冻结层
        self.freeze_bert_layers(freeze_layers)
        
        # 创建数据集
        train_dataset = PhishingTextDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if val_texts and val_labels:
            val_dataset = PhishingTextDataset(
                val_texts, val_labels, self.tokenizer, self.max_length
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 优化器和调度器
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # 训练历史
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # 训练循环
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # 计算准确率
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            avg_train_loss = total_loss / len(train_loader)
            train_acc = correct / total
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            
            log_msg = f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}"
            
            # 验证
            if val_loader:
                val_loss, val_acc = self._evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                log_msg += f" - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            
            logging.info(log_msg)
        
        logging.info("训练完成!")
        return history
    
    def _evaluate(self, data_loader) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return total_loss / len(data_loader), correct / total
    
    def predict(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        预测文本类别
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            
        Returns:
            预测标签数组
        """
        self.model.eval()
        dataset = PhishingTextDataset(texts, None, self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        预测文本类别概率
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            
        Returns:
            预测概率数组 [n_samples, 2]
        """
        self.model.eval()
        dataset = PhishingTextDataset(texts, None, self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        probabilities = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                probs = torch.softmax(outputs.logits, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        预测单个文本
        
        Args:
            text: 输入文本
            
        Returns:
            预测结果字典
        """
        proba = self.predict_proba([text])[0]
        prediction = int(np.argmax(proba))
        
        return {
            'prediction': prediction,
            'label': '钓鱼网站' if prediction == 1 else '安全网站',
            'confidence': float(proba[prediction]),
            'probabilities': {
                'benign': float(proba[0]),
                'phishing': float(proba[1])
            }
        }
    
    def save(self, path: str):
        """保存模型"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logging.info(f"模型已保存到 {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'BertPhishingClassifier':
        """加载模型"""
        instance = cls(model_path=path, device=device)
        return instance


# 测试代码
if __name__ == '__main__':
    if TRANSFORMERS_AVAILABLE:
        # 创建分类器
        classifier = BertPhishingClassifier()
        
        # 测试预测
        test_texts = [
            "Welcome to our secure banking portal. Please enter your credentials.",
            "Congratulations! You've won $1,000,000! Click here to claim your prize now!",
            "Your account has been suspended. Verify your identity immediately.",
        ]
        
        print("测试BERT分类器...")
        for text in test_texts:
            result = classifier.predict_single(text)
            print(f"文本: {text[:50]}...")
            print(f"预测: {result['label']} (置信度: {result['confidence']:.2%})")
            print()
    else:
        print("transformers未安装，跳过测试")
