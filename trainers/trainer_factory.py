"""
Trainer Factory - creates appropriate trainer for each parser type
"""

from enum import Enum
from typing import Dict, Type

from trainers.base_trainer import BaseTrainer
from trainers.bilstm_trainer import BiLSTMTrainer
from trainers.transition_trainer import TransitionTrainer


class TrainerType(Enum):
    ATTENTION = "attention"     # For all attention-based parsers (Biaffine)
    TRANSITION = "transition"   # For all transition-based parsers


class TrainerFactory:
    
    _trainers: Dict[TrainerType, Type[BaseTrainer]] = {
        TrainerType.ATTENTION: BiLSTMTrainer,
        TrainerType.TRANSITION: TransitionTrainer,
    }
    
    @classmethod
    def get_trainer_class(cls, trainer_type: TrainerType) -> Type[BaseTrainer]:
        if trainer_type not in cls._trainers:
            raise ValueError(f"Unknown trainer type: {trainer_type}")
        return cls._trainers[trainer_type]
    
    @classmethod
    def create_trainer(
        cls,
        trainer_type: TrainerType,
        model,
        train_loader,
        validation_loader,
        vocab,
        device: str,
        **kwargs
    ) -> BaseTrainer:
        trainer_class = cls.get_trainer_class(trainer_type)
        return trainer_class(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            vocab=vocab,
            device=device,
            **kwargs
        )
    
    @classmethod
    def from_parser_type(cls, parser_type: str) -> TrainerType:
        """Map parser type string to trainer type"""
        # Attention-based parsers
        attention_parsers = {
            "biaffine", "bilstm",
        }
        
        # Transition-based parsers
        transition_parsers = {
            "transition",
            "chen_manning_2014", "chen_manning",
            "weiss_2015", "weiss",
            "andor_2016", "andor",
        }
        
        parser_type_lower = parser_type.lower()
        
        if parser_type_lower in attention_parsers:
            return TrainerType.ATTENTION
        elif parser_type_lower in transition_parsers:
            return TrainerType.TRANSITION
        else:
            raise ValueError(f"Unknown parser type: {parser_type}. "
                           f"Valid attention: {attention_parsers}, "
                           f"Valid transition: {transition_parsers}")
