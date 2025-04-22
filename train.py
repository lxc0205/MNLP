import os
import torch
import shutil
import torch.nn as nn
from models.mlp import MLP
import torch.optim as optim
from models.mnlp import MNLP
from omegaconf import OmegaConf
from utils.logger import setup_logger
from torch.utils.data import DataLoader
from utils.metrics import AverageMeter, accuracy
from torch.utils.tensorboard import SummaryWriter
from data.coco_dataset import get_coco_datasets, get_coco_transform
from data.mnist_dataset import get_mnist_datasets, get_mnist_transform


def train(cfg):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化日志和保存目录
    if os.path.exists(cfg['logging']['log_dir']):
        shutil.rmtree(cfg['logging']['log_dir'])
    os.makedirs(cfg['logging']['log_dir'], exist_ok=True)
    if os.path.exists(cfg['logging']['save_dir']):
        shutil.rmtree(cfg['logging']['save_dir'])
    os.makedirs(cfg['logging']['save_dir'], exist_ok=True)
    logger = setup_logger(cfg['logging']['log_dir'])
    writer = SummaryWriter(cfg['logging']['log_dir'])
    
    if os.name == "posix": # Linux
        root_dir=cfg['data']['root_dir']['linux']
    elif os.name == "nt":  # Windows
        root_dir=cfg['data']['root_dir']['windows']
    else:
        raise ValueError("Unsupported operating system")
    
    if cfg['data']['name'] == 'mnist':
        # 数据加载
        train_transform = get_mnist_transform(train=True, image_size=cfg['data']['image_size'])
        val_transform = get_mnist_transform(train=False, image_size=cfg['data']['image_size'])

        train_dataset, val_dataset, _ = get_mnist_datasets(
            root=root_dir,
            train_transform=train_transform,
            val_transform=val_transform
        )
    if cfg['data']['name'] =='coco':
        # 数据加载
        train_transform = get_coco_transform(train=True, image_size=cfg['data']['image_size'])
        val_transform = get_coco_transform(train=False, image_size=cfg['data']['image_size'])
        
        train_dataset, val_dataset, _ = get_coco_datasets(
            root=root_dir,
            train_transform=train_transform,
            val_transform=val_transform 
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers']
    )
    
    # 模型初始化
    if cfg['model']['type'] == 'mlp':
        model = MLP(
            input_size=cfg['model']['input_size'],
            hidden_sizes=cfg['model']['hidden_sizes'],
            num_classes=cfg['model']['num_classes'],
            dropout=cfg['model']['dropout']
        ).to(device)
    elif cfg['model']['type'] == 'mnlp':    
        model = MNLP(
            input_size=cfg['model']['input_size'],
            hidden_sizes=cfg['model']['hidden_sizes'],
            num_classes=cfg['model']['num_classes'],
            dropout=cfg['model']['dropout']
        ).to(device)
    else:
        raise ValueError(f"Unsupported model name: {cfg['model']['name']}")
    
    # 损失函数和优化器 - MNIST是单标签分类
    criterion = nn.CrossEntropyLoss()  # 改为交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'], 
                          weight_decay=cfg['training']['weight_decay'])
    
    # 学习率调度器
    if cfg['training']['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=cfg['training']['lr_step_size'], 
            gamma=cfg['training']['lr_gamma']
        )
    
    best_acc = 0.0
    
    # 训练循环
    for epoch in range(cfg['training']['epochs']):
        model.train()
        losses = AverageMeter()
        acc = AverageMeter()
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录指标
            batch_acc = accuracy(outputs, labels)
            losses.update(loss.item(), images.size(0))
            acc.update(batch_acc, images.size(0))
            
            if i % 100 == 0:
                logger.info(f'Epoch: [{epoch}/{cfg["training"]["epochs"]}][{i}/{len(train_loader)}]\t'
                          f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                          f'Acc {acc.val:.3f} ({acc.avg:.3f})')
        
        # 验证阶段
        val_loss, val_acc = validate(val_loader, model, criterion, device)
        logger.info(f'Validation - Epoch: {epoch}\tLoss: {val_loss:.4f}\tAcc: {val_acc:.3f}')
        
        # 记录到TensorBoard
        writer.add_scalars('loss', {'train': losses.avg, 'val': val_loss}, epoch)
        writer.add_scalars('accuracy', {'train': acc.avg, 'val': val_acc}, epoch)
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(cfg['logging']['save_dir'], 'model_best.pth')
            print(save_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'acc': val_acc,
            }, save_path)
            logger.info(f'Saved best model to {save_path} (Acc: {val_acc:.3f})')
        
        # 定期保存
        if (epoch + 1) % cfg['logging']['save_freq'] == 0 or epoch == cfg['training']['epochs'] - 1:
            save_path = os.path.join(cfg['logging']['save_dir'], f'model_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses.avg,
                'acc': acc.avg,
            }, save_path)
            logger.info(f'Saved checkpoint to {save_path}')
    
    writer.close()

def validate(val_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    acc = AverageMeter()
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_acc = accuracy(outputs, labels)
            
            losses.update(loss.item(), images.size(0))
            acc.update(batch_acc, images.size(0))
    
    return losses.avg, acc.avg

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
        
    train(cfg)