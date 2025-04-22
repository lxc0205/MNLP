import os
import yaml
import torch
import numpy as np
import seaborn as sns
import torch.nn as nn
from models.mlp import MLP
import matplotlib.pyplot as plt
from utils.logger import setup_logger
from torch.utils.data import DataLoader
from utils.metrics import AverageMeter, accuracy
from data.mnist_dataset import get_mnist_datasets, get_transform
from sklearn.metrics import confusion_matrix, classification_report


def test(cfg, model_path):
    """
    在测试集上评估模型性能
    
    参数:
        cfg (dict): 配置字典
        model_path (str): 模型检查点路径
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化日志
    logger = setup_logger(cfg['logging']['log_dir'], name='test')
    
    # 数据加载 - 使用测试集
    test_transform = get_transform(train=False, image_size=cfg['data']['image_size'])
    _, _, test_dataset = get_mnist_datasets(
        root=cfg['data']['root_dir'],
        val_transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers']
    )
    
    # 模型加载
    model = MLP(
        input_size=cfg['model']['input_size'],
        hidden_sizes=cfg['model']['hidden_sizes'],
        num_classes=cfg['model']['num_classes'],
        dropout=0.0  # 测试时不需要dropout
    ).to(device)
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 初始化指标
    losses = AverageMeter()
    acc = AverageMeter()
    criterion = nn.CrossEntropyLoss()
    
    # 存储所有预测结果用于后续分析
    all_preds = []
    all_targets = []
    wrong_samples = []  # 存储错误样本(图像, 预测, 真实标签)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_acc = accuracy(outputs, labels)
            
            # 更新指标
            losses.update(loss.item(), images.size(0))
            acc.update(batch_acc, images.size(0))
            
            # 获取预测结果
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            # 记录错误样本
            wrong_mask = ~preds.eq(labels)
            wrong_images = images[wrong_mask]
            wrong_pred = preds[wrong_mask]
            wrong_true = labels[wrong_mask]
            
            for i in range(wrong_images.size(0)):
                if len(wrong_samples) < 25:  # 最多保存25个错误样本
                    wrong_samples.append((
                        wrong_images[i].cpu(),
                        wrong_pred[i].item(),
                        wrong_true[i].item()
                    ))
    
    # 打印总体测试结果
    logger.info(f"\nTest Results:")
    logger.info(f"Loss: {losses.avg:.4f}")
    logger.info(f"Accuracy: {acc.avg:.4f}")
    
    # 生成分类报告
    logger.info("\nClassification Report:")
    logger.info(classification_report(
        all_targets, all_preds, digits=4, zero_division=0))
    
    # 生成并保存混淆矩阵
    plot_confusion_matrix(all_targets, all_preds, 
                         save_path=os.path.join(cfg['logging']['log_dir'], 'confusion_matrix.png'))
    
    # 保存错误样本可视化
    plot_wrong_samples(wrong_samples, 
                      save_path=os.path.join(cfg['logging']['log_dir'], 'wrong_samples.png'))
    
    logger.info(f"Test results and visualizations saved to {cfg['logging']['log_dir']}")

def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    绘制并保存混淆矩阵
    
    参数:
        y_true (list): 真实标签
        y_pred (list): 预测标签
        save_path (str): 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_wrong_samples(samples, save_path, ncols=5):
    """
    绘制并保存错误分类样本
    
    参数:
        samples (list): 错误样本列表 (image, pred, true)
        save_path (str): 保存路径
        ncols (int): 每行显示的样本数
    """
    if not samples:
        return
    
    nrows = (len(samples) + ncols - 1) // ncols
    plt.figure(figsize=(2*ncols, 2*nrows))
    
    for i, (img, pred, true) in enumerate(samples):
        plt.subplot(nrows, ncols, i+1)
        img = img.squeeze().numpy()
        plt.imshow(img, cmap='gray')
        plt.title(f'P:{pred}, T:{true}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    test(cfg, args.model)