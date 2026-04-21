from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException

from core.db import _qident

# field:value where value may be "quoted string" or bare token
FIELD_VALUE_RE = re.compile(r'^[A-Za-z_]\w*:(?:"[^"]*"|\S+)$')


def _tokenize_query(q: str) -> List[str]:
    # tokens: '(', ')', 'AND', 'OR', or field:value
    q = (q or "").strip()
    if not q:
        return []
    raw = re.findall(r'\(|\)|"[^"]*"|\S+', q)
    tokens: List[str] = []
    for t in raw:
        up = t.upper()
        if up in ("AND", "OR") or t in ("(", ")"):
            tokens.append(up if up in ("AND", "OR") else t)
        else:
            tokens.append(t)
    return tokens


def _normalize_like(v: str) -> str:
    v = (v or "").strip()
    if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        v = v[1:-1]
    v = v.replace("*", "%")
    if "%" not in v:
        v = f"%{v}%"
    return v


# field -> (alias, column) for the /api/search join query
ALLOWED_FIELDS: Dict[str, Tuple[str, str]] = {
    # product (p)
    "productid": ("p", "ProductID"),
    "productdescr": ("p", "ProductDescr"),
    "productlevel": ("p", "Level"),
    "businessunit": ("p", "BusinessUnit"),
    "isdailyforecastrequired": ("p", "IsDailyForecastRequired"),
    "isnew": ("p", "IsNew"),
    "productfamily": ("p", "ProductFamily"),

    # channel (c)
    "channelid": ("c", "ChannelID"),
    "channeldescr": ("c", "ChannelDescr"),
    "channellevel": ("c", "Level"),

    # location (l)
    "locationid": ("l", "LocationID"),
    "locationdescr": ("l", "LocationDescr"),
    "locationlevel": ("l", "Level"),
    "isactive": ("l", "IsActive"),
}


def _clause_for_field_value(token: str, i: int) -> Tuple[str, Dict[str, str]]:
    field, value = token.split(":", 1)
    f = field.lower().strip()
    if f not in ALLOWED_FIELDS:
        raise HTTPException(status_code=400, detail=f"Unsupported field: {field}")

    alias, col = ALLOWED_FIELDS[f]
    pname = f"v{i}"
    clause = f'CAST({alias}.{_qident(col)} AS TEXT) ILIKE :{pname}'
    params = {pname: _normalize_like(value)}
    return clause, params


def _build_where_from_query(q: str) -> Tuple[str, Dict[str, str]]:
    """
    Grammar:
      expr  := term (OR term)*
      term  := factor (AND factor)*
      factor:= field:value | '(' expr ')'
    AND has higher precedence than OR.
    """
    tokens = _tokenize_query(q)
    if not tokens:
        return "1=1", {}

    if not any(FIELD_VALUE_RE.match(t) for t in tokens):
        raise HTTPException(
            status_code=400,
            detail='Invalid query. Use field:value (e.g., productid:*A*)'
        )

    pos = 0
    param_index = 0
    params: Dict[str, str] = {}

    def peek() -> str:
        return tokens[pos] if pos < len(tokens) else ""

    def consume(expected: Optional[str] = None) -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise HTTPException(status_code=400, detail="Unexpected end of query.")
        t = tokens[pos]
        if expected and t != expected:
            raise HTTPException(status_code=400, detail=f"Expected {expected} but found {t}")
        pos += 1
        return t

    def parse_factor() -> str:
        nonlocal param_index, params
        t = peek()
        if t == "(":
            consume("(")
            inner = parse_expr()
            if peek() != ")":
                raise HTTPException(status_code=400, detail="Missing ')'")
            consume(")")
            return f"({inner})"

        t = consume()
        if not FIELD_VALUE_RE.match(t):
            raise HTTPException(status_code=400, detail=f"Invalid token: {t}")
        clause, p = _clause_for_field_value(t, param_index)
        param_index += 1
        params.update(p)
        return clause

    def parse_term() -> str:
        left = parse_factor()
        while peek() == "AND":
            consume("AND")
            right = parse_factor()
            left = f"({left} AND {right})"
        return left

    def parse_expr() -> str:
        left = parse_term()
        while peek() == "OR":
            consume("OR")
            right = parse_term()
            left = f"({left} OR {right})"
        return left

    where_sql = parse_expr()
    if pos != len(tokens):
        raise HTTPException(status_code=400, detail=f"Unexpected token: {tokens[pos]}")
    return where_sql, params