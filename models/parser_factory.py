"""
Parser Factory - Create dependency parsers by type
"""

from enum import Enum
from typing import Dict, Type, Any, Union

from models import (
    # Traditional
    BaseTraditionalParser, MaltParser, MSTParser, TurboParser,
    # Neural Transition
    BaseTransitionParser, ChenManningParser, Weiss2015Parser, Andor2016Parser,
    # Neural Attention
    BaseAttentionParser, BiaffineAttentionParser, BertBiaffineParser
)


class ParserType(Enum):
    # Traditional parsers (non-neural)
    MALT = "malt"
    MST = "mst"
    TURBO = "turbo"
    
    # Neural transition-based parsers
    CHEN_MANNING_2014 = "chen_manning_2014"
    WEISS_2015 = "weiss_2015"
    ANDOR_2016 = "andor_2016"
    
    # Neural attention-based parsers
    BIAFFINE = "biaffine"


class ParserFactory:
    """Factory for creating dependency parsers"""
    
    _traditional: Dict[ParserType, Type[BaseTraditionalParser]] = {
        ParserType.MALT: MaltParser,
        ParserType.MST: MSTParser,
        ParserType.TURBO: TurboParser,
    }
    
    _neural_transition: Dict[ParserType, Type[BaseTransitionParser]] = {
        ParserType.CHEN_MANNING_2014: ChenManningParser,
        ParserType.WEISS_2015: Weiss2015Parser,
        ParserType.ANDOR_2016: Andor2016Parser,
    }
    
    _neural_attention: Dict[ParserType, Type[BaseAttentionParser]] = {
        ParserType.BIAFFINE: BiaffineAttentionParser,
    }
    
    @classmethod
    def get_parser_class(cls, parser_type: ParserType) -> Type:
        if parser_type in cls._traditional:
            return cls._traditional[parser_type]
        elif parser_type in cls._neural_transition:
            return cls._neural_transition[parser_type]
        elif parser_type in cls._neural_attention:
            return cls._neural_attention[parser_type]
        else:
            raise ValueError(f"Unknown parser type: {parser_type}")
    
    @classmethod
    def create_parser(cls, parser_type: ParserType, **kwargs: Any) -> Union[
        BaseTraditionalParser, BaseTransitionParser, BaseAttentionParser
    ]:
        parser_class = cls.get_parser_class(parser_type)
        return parser_class(**kwargs)
    
    @classmethod
    def from_string(cls, parser_type_str: str) -> ParserType:
        mapping = {
            # Traditional
            "malt": ParserType.MALT,
            "mst": ParserType.MST,
            "turbo": ParserType.TURBO,
            # Neural Transition
            "chen_manning_2014": ParserType.CHEN_MANNING_2014,
            "chen_manning": ParserType.CHEN_MANNING_2014,
            "weiss_2015": ParserType.WEISS_2015,
            "weiss": ParserType.WEISS_2015,
            "andor_2016": ParserType.ANDOR_2016,
            "andor": ParserType.ANDOR_2016,
            # Neural Attention
            "biaffine": ParserType.BIAFFINE,
            "bilstm": ParserType.BIAFFINE,
        }
        key = parser_type_str.lower()
        if key not in mapping:
            raise ValueError(f"Unknown parser type: {parser_type_str}. Valid: {list(mapping.keys())}")
        return mapping[key]
    
    @classmethod
    def is_traditional(cls, parser_type: ParserType) -> bool:
        return parser_type in cls._traditional
    
    @classmethod
    def is_neural_transition(cls, parser_type: ParserType) -> bool:
        return parser_type in cls._neural_transition
    
    @classmethod
    def is_neural_attention(cls, parser_type: ParserType) -> bool:
        return parser_type in cls._neural_attention
