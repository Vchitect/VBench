#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import Counter, defaultdict


ROLE_LINE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.+)\s*$")
TOKEN_RE = re.compile(r"[A-Za-z']+")

IRREGULAR = {
    "sat": "sit",
    "sits": "sit",
    "satting": "sit",
    "stood": "stand",
    "stood": "stand",
    "went": "go",
    "gone": "go",
    "took": "take",
    "taken": "take",
    "gave": "give",
    "given": "give",
    "made": "make",
    "does": "do",
    "did": "do",
    "done": "do",
    "held": "hold",
    "holds": "hold",
    "hung": "hang",
    "swung": "swing",
    "spun": "spin",
    "ran": "run",
    "run": "run",
    "lay": "lie",
    "laid": "lay",
    "lit": "light",
    "smelt": "smell",
    "swept": "sweep",
    "slept": "sleep",
    "fed": "feed",
    "felt": "feel",
    "bent": "bend",
    "left": "leave",
    "lost": "lose",
    "shot": "shoot",
    "told": "tell",
    "thought": "think",
    "kept": "keep",
    "met": "meet",
    "paid": "pay",
    "said": "say",
    "saw": "see",
    "shook": "shake",
    "sat": "sit",
    "spent": "spend",
    "stood": "stand",
    "taught": "teach",
    "threw": "throw",
    "torn": "tear",
    "wore": "wear",
    "written": "write",
    "wrote": "write",
}

AUXILIARY = {
    "be",
    "am",
    "is",
    "are",
    "was",
    "were",
    "been",
    "being",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "will",
    "would",
    "can",
    "could",
    "shall",
    "should",
    "may",
    "might",
    "must",
}

BASE_VERB_WHITELIST = {
    "adjust",
    "applaud",
    "approach",
    "attack",
    "back",
    "bend",
    "bite",
    "blink",
    "blow",
    "brush",
    "bow",
    "bring",
    "chew",
    "clap",
    "close",
    "cover",
    "cross",
    "dance",
    "drop",
    "extend",
    "exhale",
    "fan",
    "flip",
    "follow",
    "gesture",
    "giggle",
    "grab",
    "greet",
    "hold",
    "jump",
    "kick",
    "kiss",
    "laugh",
    "lean",
    "lift",
    "look",
    "lower",
    "march",
    "move",
    "nod",
    "open",
    "pat",
    "peek",
    "place",
    "point",
    "pull",
    "pump",
    "push",
    "raise",
    "reach",
    "rest",
    "rise",
    "rub",
    "run",
    "scratch",
    "shake",
    "shift",
    "shrug",
    "sit",
    "smile",
    "spin",
    "squat",
    "stand",
    "stare",
    "step",
    "stop",
    "stretch",
    "stroke",
    "swing",
    "tap",
    "tilt",
    "turn",
    "twist",
    "walk",
    "wave",
    "wipe",
    "yawn",
}


def simple_lemma(token):
    lower = token.lower()
    if lower in IRREGULAR:
        return IRREGULAR[lower]
    if lower.endswith("ies") and len(lower) > 4:
        return lower[:-3] + "y"
    if lower.endswith("ing") and len(lower) > 4:
        return lower[:-3]
    if lower.endswith("ed") and len(lower) > 3:
        return lower[:-2]
    if lower.endswith("s") and len(lower) > 3:
        return lower[:-1]
    return lower


def try_spacy(text):
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        return None
    doc = nlp(text)
    verbs = []
    for token in doc:
        if token.pos_ in {"VERB", "AUX"}:
            lemma = token.lemma_.lower()
            if lemma not in AUXILIARY:
                verbs.append(lemma)
    return verbs


def try_nltk(text):
    try:
        import nltk
        from nltk import pos_tag
        from nltk.tokenize import word_tokenize
    except Exception:
        return None
    try:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
    except Exception:
        return None
    verbs = []
    for word, tag in tagged:
        if tag.startswith("VB"):
            lemma = simple_lemma(word)
            if lemma not in AUXILIARY:
                verbs.append(lemma)
    return verbs


def heuristic_verbs(text):
    tokens = TOKEN_RE.findall(text)
    verbs = []
    for token in tokens:
        lower = token.lower()
        if lower in AUXILIARY:
            continue
        lemma = simple_lemma(lower)
        if lemma in BASE_VERB_WHITELIST:
            verbs.append(lemma)
            continue
        if lemma.endswith("ing") or lemma.endswith("ed"):
            verbs.append(lemma)
            continue
        if lower.endswith("ing") or lower.endswith("ed"):
            verbs.append(lemma)
            continue
    return verbs


def extract_verbs(text):
    verbs = try_spacy(text)
    if verbs is not None:
        return verbs
    verbs = try_nltk(text)
    if verbs is not None:
        return verbs
    return heuristic_verbs(text)


def parse_prompts(path):
    role_to_texts = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.lower().startswith("clip"):
                continue
            match = ROLE_LINE_RE.match(line)
            if not match:
                continue
            role, prompt = match.group(1), match.group(2)
            role_to_texts[role].append(prompt)
    return role_to_texts


def build_role_verbs(role_to_texts):
    role_to_verbs = {}
    role_to_counts = {}
    for role, texts in role_to_texts.items():
        counter = Counter()
        for text in texts:
            verbs = extract_verbs(text)
            counter.update(verbs)
        role_to_counts[role] = counter
        role_to_verbs[role] = [v for v, _ in counter.most_common()]
    return role_to_verbs, role_to_counts


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def write_csv(path, role_to_counts):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["role", "verb", "count"])
        for role in sorted(role_to_counts.keys()):
            for verb, count in role_to_counts[role].most_common():
                writer.writerow([role, verb, count])


def main():
    parser = argparse.ArgumentParser(description="Extract verbs per role from prompt txt.")
    parser.add_argument("--input", required=True, help="Path to prompt txt.")
    parser.add_argument("--output", help="Output path (json or csv).")
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    parser.add_argument("--print", action="store_true", help="Print a brief summary to stdout.")
    args = parser.parse_args()

    role_to_texts = parse_prompts(args.input)
    role_to_verbs, role_to_counts = build_role_verbs(role_to_texts)

    if args.output:
        if args.format == "json":
            write_json(args.output, role_to_verbs)
        else:
            write_csv(args.output, role_to_counts)

    if args.print or not args.output:
        for role in sorted(role_to_verbs.keys()):
            verbs = ", ".join(role_to_verbs[role][:20])
            print(f"{role}: {verbs}")


if __name__ == "__main__":
    main()



# python scripts/extract_verbs_from_prompts.py \
#   --input prompt_all_long.txt \
#   --output evaluation_results/role_verbs.json \
#   --format json
