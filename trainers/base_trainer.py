import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.logs import train_logger
from data import Vocabulary
from evaluation import Evaluator


class BaseTrainer(ABC):
    
    def __init__(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        validation_loader: DataLoader, 
        vocab: Vocabulary, 
        device: str, 
        lr: float = 2e-3, 
        weight_decay: float = 1e-4, 
        save_dir: str = 'checkpoints',
        save_every: int = 5
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.vocab = vocab
        self.device = device
        self.save_dir = save_dir
        self.save_every = save_every
        
        self.evaluator = Evaluator(vocab=vocab)
        
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay, 
            betas=(0.9, 0.9)
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
        
        self.best_uas = 0.0
        self.best_las = 0.0
        self.best_epoch = 0
        
        self.history: List[Dict] = []
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    @abstractmethod
    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """
        Train one epoch.
        
        Returns:
            (total_loss, arc_loss, label_loss)
        """
        raise NotImplementedError
    
    def evaluate(self, data_loader: Optional[DataLoader] = None) -> Dict:
        if data_loader is None:
            data_loader = self.validation_loader
        
        return self.evaluator.evaluate_model(
            self.model, 
            data_loader, 
            device=self.device
        )
    
    def train(self, num_epochs: int) -> List[Dict]:
        train_logger.info(f"Starting training for {num_epochs} epochs")
        train_logger.info(f"Device: {self.device}")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            avg_loss, avg_arc_loss, avg_label_loss = self.train_epoch(epoch)
            
            results = self.evaluate()
            dev_uas = results['uas']
            dev_las = results['las']
            
            self.scheduler.step(dev_las)
            
            epoch_time = time.time() - start_time
            
            train_logger.info(
                f"Epoch {epoch}/{num_epochs} | "
                f"Loss: {avg_loss:.4f} (arc: {avg_arc_loss:.4f}, lbl: {avg_label_loss:.4f}) | "
                f"UAS: {dev_uas:.2f}% | LAS: {dev_las:.2f}% | "
                f"Time: {epoch_time:.1f}s"
            )
            
            self.history.append({
                'epoch': epoch,
                'train_loss': avg_loss,
                'train_arc_loss': avg_arc_loss,
                'train_label_loss': avg_label_loss,
                'dev_uas': dev_uas,
                'dev_las': dev_las,
                'time': epoch_time
            })
            
            is_best = dev_las > self.best_las
            if is_best:
                self.best_las = dev_las
                self.best_uas = dev_uas
                self.best_epoch = epoch
                train_logger.info(f"New best model! LAS: {dev_las:.2f}%")
            
            self.save(epoch, is_best)
        
        train_logger.info(
            f"Training completed! Best: Epoch {self.best_epoch} | "
            f"UAS: {self.best_uas:.2f}% | LAS: {self.best_las:.2f}%"
        )
        
        return self.history
    
    def save(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_uas': self.best_uas,
            'best_las': self.best_las,
            'history': self.history
        }
        
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
        
        if epoch % self.save_every == 0:
            checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
            train_logger.info(f"Saved checkpoint: epoch {epoch}")
    
    def load(self, checkpoint_path: str) -> int:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_uas = checkpoint.get('best_uas', 0.0)
        self.best_las = checkpoint.get('best_las', 0.0)
        self.history = checkpoint.get('history', [])
        
        train_logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return checkpoint.get('epoch', 0)
