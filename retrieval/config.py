from __future__ import annotations

import os

from core.config import (
    BGE_QUERY_PREFIX,
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
    DEFAULT_DENSE_WEIGHT,
    DEFAULT_EXACT_KEYWORD_FIRST,
    DEFAULT_FETCH_K,
    DEFAULT_KEYWORD_FETCH_K,
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_MAX_DISTANCE,
    DEFAULT_MAX_KEYWORDS,
    DEFAULT_METADATA_WEIGHT,
    DEFAULT_MIN_HYBRID_SCORE,
    DEFAULT_POSITION_WEIGHT,
    DEFAULT_RELATION_WEIGHT,
    DEFAULT_SUMMARY_WEIGHT,
    DEFAULT_TOP_K,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_CONCURRENCY,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_MODEL,
    EMBEDDING_REQUEST_TIMEOUT,
    MAX_DISTANCE,
    MIN_KNOWN_PERSON_FREQUENCY,
    PROCESSED_DIR,
)

# 配置已收口至 core.config(上方导入后再导出);此处仅保留派生常量与
# SILICONFLOW embedding 凭证(凭证不集中,见 core/config.py 模块说明)。
RELATIONS_PATH = PROCESSED_DIR / "relations.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

EMBEDDING_API_BASE = os.getenv("SILICONFLOW_API_BASE", "https://api.siliconflow.cn/v1")
EMBEDDING_API_KEY = os.getenv("SILICONFLOW_API_KEY")

COLLECTION_METADATA = {
    "description": "Local StoryRAG knowledge base",
    "embedding_model": EMBEDDING_MODEL,
}
COLLECTION_CONFIGURATION = {"hnsw": {"space": "cosine"}}


QUESTION_PHRASES = (
    "请问",
    "告诉我",
    "介绍一下",
    "解释一下",
    "是什么",
    "是谁",
    "在哪里",
    "为什么",
    "怎么样",
    "怎么",
    "如何",
    "多少",
    "哪一个",
    "哪位",
    "哪个",
    "什么",
    "吗",
    "么",
    "呢",
    "呀",
    "啊",
)

STOP_KEYWORDS = {
    "请问",
    "告诉我",
    "介绍",
    "介绍一下",
    "解释",
    "解释一下",
    "一下",
    "是什么",
    "是谁",
    "怎么",
    "怎么样",
    "如何",
    "为什么",
    "什么",
    "哪个",
    "哪位",
    "哪一个",
    "多少",
    "哪里",
    "在哪里",
    "相关",
    "有关",
    "内容",
    "情况",
    "问题",
    "答案",
    "吗",
    "么",
    "呢",
    "呀",
}

QUERY_PERSON_STOPWORDS = STOP_KEYWORDS | {
    "关系",
    "人物",
    "角色",
    "剧情",
    "章节",
    "地点",
    "事件",
    "故事",
    "小说",
    "内容",
    "结局",
    "老师",
    "教授",
    "导师",
    "敌人",
    "朋友",
    "家人",
}

RELATION_TYPE_HINTS = {
    "family": {"家人", "亲人", "父亲", "母亲", "兄弟", "姐妹", "夫妻", "亲属"},
    "friend": {"朋友", "友情", "好友", "同伴"},
    "mentor": {"老师", "教授", "导师", "师父", "指导"},
    "enemy": {"敌人", "仇人", "对手"},
    "helper": {"帮助", "帮忙", "保护", "援助"},
}

RELATION_QUERY_HINTS = {
    "关系",
    "冲突",
    "敌对",
    "朋友",
    "家人",
    "老师",
    "教授",
    "导师",
    "帮助",
    "仇人",
    "对手",
    "矛盾",
    "针对",
    "讨厌",
    "保护",
    "亲属",
}

RELATION_INTENT_PATTERNS = {
    "enemy": ("冲突", "敌对", "仇人", "对手", "矛盾", "针对", "讨厌"),
    "mentor": ("老师", "教授", "导师", "指导", "教导"),
    "family": ("家人", "亲人", "父亲", "母亲", "兄弟", "姐妹", "亲属"),
    "friend": ("朋友", "友情", "好友", "同伴"),
    "helper": ("帮助", "帮忙", "保护", "救助", "援助"),
}

LOCATION_SUFFIXES = (
    "路",
    "街",
    "巷",
    "村",
    "镇",
    "城",
    "市",
    "省",
    "国",
    "学校",
    "学院",
    "餐厅",
    "图书馆",
    "教室",
    "医院",
    "车站",
    "庄园",
    "城堡",
    "办公室",
    "宿舍",
)

OBJECT_HINTS = (
    "魔法石",
    "魔杖",
    "扫帚",
    "地图",
    "日记",
    "项链",
    "戒指",
    "钥匙",
    "信",
    "斗篷",
    "长袍",
    "眼镜",
)

EVENT_HINTS = (
    "遇见",
    "收到",
    "发现",
    "进入",
    "离开",
    "前往",
    "逃离",
    "战斗",
    "袭击",
    "死亡",
    "出生",
    "出现",
    "救下",
    "阻止",
    "帮助",
    "争吵",
)

PERSON_TITLE_SUFFIXES = ("教授", "老师", "导师", "先生", "女士", "夫人", "小姐", "同学", "校长")
PERSON_QUERY_NOISE = (
    "第一",
    "第二",
    "第三",
    "出现",
    "冲突",
    "关系",
    "什么",
    "哪个",
    "哪里",
    "章节",
    "在哪",
    "一次",
    "想",
    "叫",
    "吗",
    "呢",
    "啊",
)

PERSON_NAME_BLACKLIST = {
    "第一",
    "第二",
    "第三",
    "朋友",
    "男孩",
    "女孩",
    "教授",
    "老师",
    "导师",
    "先生",
    "女士",
    "夫人",
    "小姐",
    "校长",
    "是吗",
    "谢谢",
    "当然",
    "终于",
    "怎么",
    "真的",
}

