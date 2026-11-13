"""
TwinQL Parser - Grammar specification with formal semantics
"""
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

# ============================================================================
# TwinQL Grammar (BNF)
# ============================================================================
"""
<twinql_stmt>    ::= <define_scenario> | <twin_select>

<define_scenario> ::= "DEFINE" "SCENARIO" <identifier> "AS" <scenario_body>
<scenario_body>   ::= <scenario_attr> ("," <scenario_attr>)*
<scenario_attr>   ::= <attr_name> "=" <attr_value>

<twin_select>    ::= "TWIN" "SELECT" <select_list>
                     "FROM" <from_clause>
                     ["WHERE" <predicate>]
                     ["GROUP" "BY" <group_expr>]

<from_clause>    ::= <source> ("," <source>)*
<source>         ::= <historical> | <simulate>

<historical>     ::= "HISTORICAL" "twin_id" "=" <string>
                     "WINDOW" <string> "TO" <string>
                     "METRIC" <string>
                     ["AGGREGATE" "BY" <time_unit>] "AS" <alias>

<simulate>       ::= "SIMULATE" "twin_id" "=" <string>
                     "SCENARIO" <identifier>
                     "MODEL" <string>
                     "WINDOW" <string> "TO" <string>
                     "METRIC" <string>
                     ["AGGREGATE" "BY" <time_unit>] "AS" <alias>

<time_unit>      ::= "month" | "day" | "hour" | "year"
<agg_func>       ::= "sum" | "avg" | "max" | "min" | "count"
"""

class TokenType(Enum):
    # Keywords
    DEFINE = "DEFINE"
    SCENARIO = "SCENARIO"
    AS = "AS"
    TWIN = "TWIN"
    SELECT = "SELECT"
    FROM = "FROM"
    WHERE = "WHERE"
    GROUP = "GROUP"
    BY = "BY"
    HISTORICAL = "HISTORICAL"
    SIMULATE = "SIMULATE"
    WINDOW = "WINDOW"
    TO = "TO"
    METRIC = "METRIC"
    MODEL = "MODEL"
    AGGREGATE = "AGGREGATE"
    
    # Literals
    IDENTIFIER = "IDENTIFIER"
    STRING = "STRING"
    NUMBER = "NUMBER"
    JSON = "JSON"
    
    # Operators
    EQUALS = "="
    COMMA = ","
    LPAREN = "("
    RPAREN = ")"
    DOT = "."
    MINUS = "-"
    
    # Special
    EOF = "EOF"

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int

class Lexer:
    """Tokenizer for TwinQL"""
    
    KEYWORDS = {
        'DEFINE', 'SCENARIO', 'AS', 'TWIN', 'SELECT', 'FROM', 'WHERE',
        'GROUP', 'BY', 'HISTORICAL', 'SIMULATE', 'WINDOW', 'TO',
        'METRIC', 'MODEL', 'AGGREGATE', 'USING'
    }
    
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.col = 1
    
    def tokenize(self) -> List[Token]:
        tokens = []
        while self.pos < len(self.text):
            self._skip_whitespace()
            if self.pos >= len(self.text):
                break
            
            ch = self.text[self.pos]
            
            if ch == "'":
                tokens.append(self._read_string())
            elif ch == '"':
                tokens.append(self._read_string('"'))
            elif ch == '=':
                tokens.append(Token(TokenType.EQUALS, '=', self.line, self.col))
                self._advance()
            elif ch == ',':
                tokens.append(Token(TokenType.COMMA, ',', self.line, self.col))
                self._advance()
            elif ch == '(':
                tokens.append(Token(TokenType.LPAREN, '(', self.line, self.col))
                self._advance()
            elif ch == ')':
                tokens.append(Token(TokenType.RPAREN, ')', self.line, self.col))
                self._advance()
            elif ch == '.':
                tokens.append(Token(TokenType.DOT, '.', self.line, self.col))
                self._advance()
            elif ch == '-':
                tokens.append(Token(TokenType.MINUS, '-', self.line, self.col))
                self._advance()
            elif ch.isalpha() or ch == '_':
                tokens.append(self._read_identifier())
            elif ch.isdigit():
                tokens.append(self._read_number())
            elif ch == '{':
                tokens.append(self._read_json())
            else:
                self._advance()
        
        tokens.append(Token(TokenType.EOF, '', self.line, self.col))
        return tokens
    
    def _advance(self):
        if self.pos < len(self.text):
            if self.text[self.pos] == '\n':
                self.line += 1
                self.col = 1
            else:
                self.col += 1
            self.pos += 1
    
    def _skip_whitespace(self):
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self._advance()
    
    def _read_string(self, quote="'") -> Token:
        start_col = self.col
        self._advance()  # skip opening quote
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] != quote:
            self._advance()
        value = self.text[start:self.pos]
        self._advance()  # skip closing quote
        return Token(TokenType.STRING, value, self.line, start_col)
    
    def _read_identifier(self) -> Token:
        start_col = self.col
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
            self._advance()
        value = self.text[start:self.pos]
        
        if value.upper() in self.KEYWORDS:
            return Token(TokenType[value.upper()], value, self.line, start_col)
        return Token(TokenType.IDENTIFIER, value, self.line, start_col)
    
    def _read_number(self) -> Token:
        start_col = self.col
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isdigit() or self.text[self.pos] == '.'):
            self._advance()
        return Token(TokenType.NUMBER, self.text[start:self.pos], self.line, start_col)
    
    def _read_json(self) -> Token:
        start_col = self.col
        start = self.pos
        depth = 0
        while self.pos < len(self.text):
            if self.text[self.pos] == '{':
                depth += 1
            elif self.text[self.pos] == '}':
                depth -= 1
                if depth == 0:
                    self._advance()
                    break
            self._advance()
        return Token(TokenType.JSON, self.text[start:self.pos], self.line, start_col)


@dataclass
class ScenarioDefAST:
    """AST node for DEFINE SCENARIO"""
    name: str
    config: Dict[str, Any]

@dataclass
class HistoricalAST:
    """AST node for HISTORICAL clause"""
    twin_id: str
    window_start: str
    window_end: str
    metric: str
    agg_by: Optional[str]
    agg_func: str
    alias: str

@dataclass
class SimulateAST:
    """AST node for SIMULATE clause"""
    twin_id: str
    scenario: str
    model: str
    window_start: str
    window_end: str
    metric: str
    agg_by: Optional[str]
    agg_func: str
    alias: str

@dataclass
class TwinSelectAST:
    """AST node for TWIN SELECT"""
    select_list: List[str]
    sources: List[Any]  # HistoricalAST or SimulateAST
    where_clause: Optional[str]
    group_by: Optional[str]


class Parser:
    """Parser for TwinQL"""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
    
    def parse(self) -> Any:
        """Parse a TwinQL statement"""
        if self._check(TokenType.DEFINE):
            return self._parse_define_scenario()
        elif self._check(TokenType.TWIN):
            return self._parse_twin_select()
        else:
            raise SyntaxError(f"Unexpected token: {self._current().value}")
    
    def _current(self) -> Token:
        return self.tokens[self.pos]
    
    def _check(self, type: TokenType) -> bool:
        return self._current().type == type
    
    def _advance(self) -> Token:
        token = self._current()
        if not self._check(TokenType.EOF):
            self.pos += 1
        return token
    
    def _expect(self, type: TokenType) -> Token:
        if not self._check(type):
            raise SyntaxError(f"Expected {type}, got {self._current().type}")
        return self._advance()
    
    def _parse_define_scenario(self) -> ScenarioDefAST:
        self._expect(TokenType.DEFINE)
        self._expect(TokenType.SCENARIO)
        name = self._expect(TokenType.IDENTIFIER).value
        self._expect(TokenType.AS)
        
        config = {}
        while not self._check(TokenType.EOF) and not self._check(TokenType.TWIN):
            key = self._expect(TokenType.IDENTIFIER).value
            self._expect(TokenType.EQUALS)
            
            if self._check(TokenType.STRING):
                value = self._advance().value
            elif self._check(TokenType.JSON):
                import json
                value = json.loads(self._advance().value)
            else:
                value = self._advance().value
            
            config[key.lower()] = value
            
            if self._check(TokenType.COMMA):
                self._advance()
        
        return ScenarioDefAST(name=name, config=config)
    
    def _parse_twin_select(self) -> TwinSelectAST:
        self._expect(TokenType.TWIN)
        self._expect(TokenType.SELECT)
        
        # Parse select list (simplified)
        select_list = []
        while not self._check(TokenType.FROM):
            select_list.append(self._advance().value)
            if self._check(TokenType.COMMA):
                self._advance()
        
        self._expect(TokenType.FROM)
        
        # Parse sources
        sources = []
        while not self._check(TokenType.EOF) and not self._check(TokenType.WHERE) and not self._check(TokenType.GROUP):
            if self._check(TokenType.HISTORICAL):
                sources.append(self._parse_historical())
            elif self._check(TokenType.SIMULATE):
                sources.append(self._parse_simulate())
            elif self._check(TokenType.COMMA):
                self._advance()
            else:
                break
        
        return TwinSelectAST(
            select_list=select_list,
            sources=sources,
            where_clause=None,
            group_by=None
        )
    
    def _parse_historical(self) -> HistoricalAST:
        self._expect(TokenType.HISTORICAL)
        self._expect(TokenType.IDENTIFIER)  # twin_id
        self._expect(TokenType.EQUALS)
        twin_id = self._expect(TokenType.STRING).value
        
        self._expect(TokenType.WINDOW)
        window_start = self._expect(TokenType.STRING).value
        self._expect(TokenType.TO)
        window_end = self._expect(TokenType.STRING).value
        
        self._expect(TokenType.METRIC)
        metric = self._expect(TokenType.STRING).value
        
        agg_by = None
        agg_func = 'sum'
        if self._check(TokenType.AGGREGATE):
            self._advance()
            self._expect(TokenType.BY)
            agg_by = self._expect(TokenType.IDENTIFIER).value
        
        self._expect(TokenType.AS)
        alias = self._expect(TokenType.IDENTIFIER).value
        
        return HistoricalAST(
            twin_id=twin_id,
            window_start=window_start,
            window_end=window_end,
            metric=metric,
            agg_by=agg_by,
            agg_func=agg_func,
            alias=alias
        )
    
    def _parse_simulate(self) -> SimulateAST:
        self._expect(TokenType.SIMULATE)
        self._expect(TokenType.IDENTIFIER)  # twin_id
        self._expect(TokenType.EQUALS)
        twin_id = self._expect(TokenType.STRING).value
        
        self._expect(TokenType.SCENARIO)
        scenario = self._expect(TokenType.IDENTIFIER).value
        
        self._expect(TokenType.MODEL)
        model = self._expect(TokenType.STRING).value
        
        self._expect(TokenType.WINDOW)
        window_start = self._expect(TokenType.STRING).value
        self._expect(TokenType.TO)
        window_end = self._expect(TokenType.STRING).value
        
        self._expect(TokenType.METRIC)
        metric = self._expect(TokenType.STRING).value
        
        agg_by = None
        agg_func = 'sum'
        if self._check(TokenType.AGGREGATE):
            self._advance()
            self._expect(TokenType.BY)
            agg_by = self._expect(TokenType.IDENTIFIER).value
        
        self._expect(TokenType.AS)
        alias = self._expect(TokenType.IDENTIFIER).value
        
        return SimulateAST(
            twin_id=twin_id,
            scenario=scenario,
            model=model,
            window_start=window_start,
            window_end=window_end,
            metric=metric,
            agg_by=agg_by,
            agg_func=agg_func,
            alias=alias
        )


def parse_twinql(query: str) -> Any:
    """Parse a TwinQL query string into AST"""
    lexer = Lexer(query)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()
