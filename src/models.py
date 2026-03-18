from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class ArticleBase:
    title: str
    link: str
    pub_date: datetime
    source: str


@dataclass
class Article(ArticleBase):
    content: str


@dataclass
class CleanedArticle(Article):
    id: int


@dataclass
class ProcessedArticle(ArticleBase):
    id: int
    one_line: str
    category: str


@dataclass
class ProcessedResult:
    articles: List[ProcessedArticle]
    categories: List[str]
    summary_lines: List[str]
    degraded: bool = False
    warnings: List[str] = field(default_factory=list)
