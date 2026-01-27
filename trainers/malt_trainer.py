from pathlib import Path
from typing import Union
from datetime import datetime

import torch
import torch.distributed as dist

from utils.logs import logger
from utils.sp_data import Dataset
from utils.sp_field import Field
from utils.constants import pad, unk, bos
from utils.sp_metric import Metric, AttachmentMetric

from transforms.conll import CoNLL, progress_bar
from models.transition_based.malt_parser import MaltParser
from datasets import ViVTBCorpus

class MaltTrainer:
    def __init__(self, parser: MaltParser, corpus: ViVTBCorpus):
        self.parser = parser
        self.corpus = corpus

    def train(self, 
        base_path: Union[Path, str],
        fix_len=20,
        min_freq=2,
        batch_size: int = 5000,
        buckets=1000, 
        max_epochs: int = 10, 
        wandb=None
    ):
        ARC = Field('arcs', bos=bos, use_vocab=False, fn=CoNLL.get_arcs)
        REL = Field('rels', bos=bos)
        WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
        FEAT = Field('tags', bos=bos)
        transform = CoNLL(FORM=WORD, CPOS=FEAT, HEAD=ARC, DEPREL=REL)

        train = Dataset(transform, self.corpus.train)
        WORD.build(train, min_freq)
        FEAT.build(train)
        REL.build(train)

        parser = MaltParser(
            transform=transform
        )
        if wandb:
            wandb.watch(parser)


        logger.info('Loading the data')
        train = Dataset(parser.transform, self.corpus.train)
        dev = Dataset(parser.transform, self.corpus.dev)
        test = Dataset(parser.transform, self.corpus.test)
        train.build(batch_size, buckets, True, dist.is_initialized())
        dev.build(batch_size, buckets)
        test.build(batch_size, buckets)

        logger.info(f"\n{'train:':6} {train}\n{'dev:':6} {dev}\n{'test:':6} {test}\n")
        logger.info(f'{parser}')

        for epoch in range(1, max_epochs + 1):
            start = datetime.now()
            logger.info(f'Epoch {epoch} / {max_epochs}:')

            bar = progress_bar(train.loader)
            metric = AttachmentMetric()
            for words, feats, arcs, rels in bar:
                mask = words.ne(parser.WORD.pad_index) if parser.WORD else torch.ones_like(words, dtype=torch.bool)
                # ignore the first token of each sentence
                mask[:, 0] = 0 
                
                sentences = words.shape[0]
                for i in range(sentences):
                    # Lấy độ dài thực (bao gồm cả ROOT)
                    length = mask[i].sum().item() + 1
                    
                    # Convert to list & remove padding
                    sent_words = words[i, :length].tolist()
                    sent_feats = feats[i, :length].tolist()
                    sent_heads = arcs[i, :length].tolist()
                    sent_rels = rels[i, :length].tolist()
                    
                    # Update (Train) Parser
                    parser.update(sent_words, sent_feats, sent_heads, sent_rels)
                    
            
            # End of epoch logging
            logger.info(f"Epoch {epoch} finished. Total steps: {parser.steps}")
            
            # === Evaluate on Dev ===
            logger.info("Evaluating on Dev...")
            metric_dev = AttachmentMetric()
            # Disable progress bar for dev to keep log clean or enable if prefer
            bar = progress_bar(dev.loader)
            for words, feats, arcs, rels in bar:
                mask = words.ne(parser.WORD.pad_index) if parser.WORD else torch.ones_like(words, dtype=torch.bool)
                mask[:, 0] = 0
                
                # Init empty tensors for prediction
                pred_arcs = torch.zeros_like(arcs)
                pred_rels = torch.zeros_like(rels) # Cần mapping str->id nếu muốn tính LAS chuẩn
                
                real_batch_size = words.shape[0]
                for i in range(real_batch_size):
                    length = mask[i].sum().item() + 1
                    sent_words = words[i, :length].tolist()
                    sent_feats = feats[i, :length].tolist()
                    
                    # Inference
                    p_arcs, p_rels = parser.parse(sent_words, sent_feats)
                    
                    # Fill Tensor
                    for dep, head in p_arcs.items():
                        if dep < length:
                            pred_arcs[i, dep] = head
                            
                    for dep, label_str in p_rels.items():
                        if dep < length:
                             # Mapping Str -> ID
                             # Nếu label lạ ko có trong vocab, lấy UNK (thường là 0)
                             rel_id = parser.vocab_rels.stoi.get(label_str, 0)
                             pred_rels[i, dep] = rel_id
                
                # Calculate Metric
                metric_dev(pred_arcs, pred_rels, arcs, rels, mask)
                
            logger.info(f"Dev Results: {metric_dev}") 
        
        logger.info("Evaluating on Test...")
        metric_test = AttachmentMetric()
        bar = progress_bar(test.loader)
        for words, feats, arcs, rels in bar:
            mask = words.ne(parser.WORD.pad_index) if parser.WORD else torch.ones_like(words, dtype=torch.bool)
            mask[:, 0] = 0
            
            # Init empty tensors for prediction
            pred_arcs = torch.zeros_like(arcs)
            pred_rels = torch.zeros_like(rels) # Cần mapping str->id nếu muốn tính LAS chuẩn
            
            real_batch_size = words.shape[0]
            for i in range(real_batch_size):
                length = mask[i].sum().item() + 1
                sent_words = words[i, :length].tolist()
                sent_feats = feats[i, :length].tolist()
                
                # Inference
                p_arcs, p_rels = parser.parse(sent_words, sent_feats)
                
                # Fill Tensor
                for dep, head in p_arcs.items():
                    if dep < length:
                        pred_arcs[i, dep] = head
                        
                for dep, label_str in p_rels.items():
                    if dep < length:
                         # Mapping Str -> ID
                         # Nếu label lạ ko có trong vocab, lấy UNK (thường là 0)
                         rel_id = parser.vocab_rels.stoi.get(label_str, 0)
                         pred_rels[i, dep] = rel_id
                
            # Calculate Metric
            metric_test(pred_arcs, pred_rels, arcs, rels, mask)
            
        logger.info(f"Test Results: {metric_test}")
        
        # Save model using pickle
        import pickle
        import os
        os.makedirs(os.path.dirname(str(base_path)) or '.', exist_ok=True)
        save_path = str(base_path) if str(base_path).endswith('.pkl') else f"{base_path}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(parser, f)
        logger.info(f"Model saved to: {save_path}") 