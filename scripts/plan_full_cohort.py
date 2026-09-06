"""Annotation-only review queues. Text similarity NEVER establishes video identity.

No model predictions are loaded. Queue membership is provisional until media,
viewpoint, evidence, donors, and cross-split constituent overlap are audited.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

PREVIOUS_TEST = {f"test:positional_relationship_{value}" for value in ("0002", "0055", "0088", "0103", "0188")}
ATTRIBUTES = set("red green blue white black gray grey brown yellow orange purple pink beige silver golden gold "
                 "dark light colored colour colors colours toned square circular round rectangular triangular "
                 "trapezoidal oval arc semi semicircular shaped spherical cylindrical irregular".split())
STOP = set("the a an of with and in on at to for is are was were what which located consists multiple "
           "concatenated clips clip video first second third fourth fifth this that there it its".split())
FAMILIES = {
    "sports": r"court|stadium|football|basketball|tennis|track|runway|playing field|running",
    "water": r"pool|fountain|lake|river|canal|sea|ocean|water|shore|beach",
    "transport": r"road|highway|bridge|rail|train|boat|ship|vehicle|bus|car|truck|motorcycle",
    "vegetation": r"tree|forest|mountain|hill|grass|hedge|vegetation|park|field|farmland",
    "structure": r"building|roof|pavilion|tower|lighthouse|skyscraper|wall|gate|chimney|fence|house|sculpture",
    "sign": r"sign|billboard|logo|text|\d|[\u3400-\u9fff]",
}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def normalize(text):
    return " ".join(re.findall(r"[\w]+", text.lower()))


def words(text):
    return [word for word in normalize(text).split() if word not in STOP]


def stable(value, seed):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def family(text):
    return next((name for name, pattern in FAMILIES.items() if re.search(pattern, text, re.I)), "other")


def anchor(question):
    match = re.search(r"located\s+.+?\s+of\s+(?:the\s+)?(.+?)\??$", question, re.I)
    return match.group(1).rstrip(" ?") if match else re.sub(r"^.*?clip,\s*", "", question, flags=re.I).rstrip(" ?")


def classify(record):
    question = record["question"]
    options = list(record["options"].values())
    ocr = sum(bool(re.search(r"[\u3400-\u9fff]|\d|\bDJ[It]\b", text, re.I)) for text in options) >= 2
    cores = [" ".join(word for word in words(text) if word not in ATTRIBUTES) for text in options]
    pairs = [core for core, count in Counter(cores).items() if core and count >= 2]
    kind = "ocr_numeric_symbol_choices" if ocr else "attribute_contrast_choices" if pairs else "object_choices"
    depth = bool(re.search(r"\b(?:rear|front|behind|travel)\b", question, re.I))
    relation = "viewpoint_review" if depth else "left" if re.search(r"\bleft\b", question, re.I) else "right" if re.search(r"\bright\b", question, re.I) else "vertical_other"
    target = anchor(question)
    detailed = " ".join(words(target))
    # This is a duplicate-review cue, deliberately NOT a source_group assignment.
    cue = detailed if len(words(target)) >= 4 else normalize(question) + "|" + "|".join(sorted(map(normalize, options)))
    return {"choice_kind": kind, "option_attribute_contrast_cores": pairs,
            "anchor_text_reference": bool(re.search(r"[\u3400-\u9fff]|\b(?:sign|logo|text)\b", target, re.I)),
            "relation_family": relation, "anchor": target, "anchor_family": family(target),
            "text_duplicate_review_cue": cue,
            "priority_tier": 2 if ocr else 1 if depth else 0}


def diverse_queue(rows, seed):
    result = []
    for tier in range(3):
        groups = defaultdict(list)
        for row in rows:
            p = row["cohort_planning"]
            if p["priority_tier"] == tier:
                key = (p["choice_kind"], p["anchor_family"], p["relation_family"], row.get("target_clip"))
                groups[key].append(row)
        for group in groups.values():
            group.sort(key=lambda row: stable(row["id"], seed))
        keys = sorted(groups, key=lambda key: stable(str(key), seed))
        seen = set()
        deferred = []
        while any(groups.values()):
            for key in keys:
                if not groups[key]:
                    continue
                row = groups[key].pop(0)
                cue = row["cohort_planning"]["text_duplicate_review_cue"]
                if cue in seen:
                    deferred.append(row)
                else:
                    result.append(row)
                    seen.add(cue)
        result.extend(deferred)
    for index, row in enumerate(result, 1):
        row["cohort_planning"]["priority_rank"] = index
    return result


def summaries(rows):
    appearance, correct = Counter(), Counter()
    for row in rows:
        appearance.update(re.sub(r"^(?:the|a|an) ", "", normalize(value)) for value in row["options"].values())
        correct.update([re.sub(r"^(?:the|a|an) ", "", normalize(row["options"][row["answer"]]))])
    return {"questions": len(rows), "labels": dict(Counter(row["answer"] for row in rows)),
            "choice_kinds": dict(Counter(classify(row)["choice_kind"] for row in rows)),
            "relation_families": dict(Counter(classify(row)["relation_family"] for row in rows)),
            "anchor_text_reference": sum(classify(row)["anchor_text_reference"] for row in rows),
            "frequent_options": [{"text": text, "appearances": count, "correct": correct[text]}
                                 for text, count in appearance.most_common(20)]}


def positive_annotation_text(row):
    question = row.get("question", "")
    value = row.get("answer", "")
    options = row.get("options")
    if isinstance(options, dict):
        value = options.get(value, "")
    # Negative existence options and future predictions are not presence evidence.
    if re.search(r"\bnot\b|never|except|incorrect|future|next|predict", question, re.I):
        value = ""
    if row.get("task_type") == "object_existence":
        value = ""
    return question + " " + (value if isinstance(value, str) else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="data/cohort_plan")
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--skip-auxiliary", action="store_true")
    args = parser.parse_args()
    data, output = Path(args.data_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train = read_rows(data / "normalized/train.jsonl")
    test = read_rows(data / "normalized/test.jsonl")
    blocked_media = {row["source_path"] for row in test if row["id"] in PREVIOUS_TEST}
    excluded = [row for row in test if row["id"] in PREVIOUS_TEST or row["source_path"] in blocked_media]
    candidates = [{**row, "cohort_planning": classify(row)} for row in test
                  if row["is_earlier_candidate"] and row["id"] not in PREVIOUS_TEST
                  and row["source_path"] not in blocked_media]
    queue = diverse_queue(candidates, args.seed)
    write_rows(output / "test_priority.jsonl", queue)
    write_rows(output / "first50_review_candidates.jsonl", queue[:50])
    write_rows(output / "excluded_prior_test.jsonl", excluded)
    write_rows(output / "train_question_metadata.jsonl",
               [{"id": row["id"], "source_path": row["source_path"], **classify(row)} for row in train])
    report = {
        "status": "annotation-only provisional acquisition/review queue; not a frozen test cohort",
        "seed": args.seed,
        "selection_rule": "Tier0 image-plane/nonOCR; tier1 other nonOCR viewpoints; tier2 OCR/numeric/symbol. "
                          "Round-robin choice-kind/anchor-family/relation/target-clip strata with SHA256 ordering; "
                          "defer repeated detailed text-anchor cues within each tier. No predictions read.",
        "source_identity_rule": "Text overlap is a review hint only. Do not assign source groups from it. "
                                "Exclude confirmed constituent matches to all five prior test videos after fingerprinting.",
        "train_all": summaries(train), "test_all": summaries(test),
        "test_nonfinal": summaries([row for row in test if row["is_earlier_candidate"]]),
        "test_remaining_queue": summaries(queue), "first50_provisional": summaries(queue[:50]),
        "queue_tiers": dict(Counter(row["cohort_planning"]["priority_tier"] for row in queue)),
        "excluded_ids": [row["id"] for row in excluded], "excluded_constructed_media": sorted(blocked_media),
        "queue_sha256": hashlib.sha256((output / "test_priority.jsonl").read_bytes()).hexdigest(),
        "input_provenance": {
            name: {"sha256": hashlib.sha256((data / "normalized" / f"{name}.jsonl").read_bytes()).hexdigest(),
                   "dataset_revisions": sorted({row.get("dataset_revision", "unknown") for row in rows})}
            for name, rows in (("train", train), ("test", test))
        },
        "taxonomy_note": "Attribute contrast means at least two options collapse after explicit color/shape-word "
                         "removal; it does not prove tiny details are needed. OCR bucket is a reproducible textual "
                         "screen (2+ options contain CJK/digits/DJI-like symbols), not visual ground truth.",
    }
    (output / "counts_and_rules.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"queue": str(output / "test_priority.jsonl"), "sha256": report["queue_sha256"],
                      "tiers": report["queue_tiers"], "first50": [row["id"] for row in queue[:50]]}), flush=True)
    if args.skip_auxiliary:
        return

    # Index all released annotation questions and plausible positive answers.
    # Only look for detailed anchor phrases; options never establish presence.
    grams_wanted = set()
    for row in queue + train:
        tokens = words(anchor(row["question"]))
        grams_wanted.update(tuple(tokens[i:i + 4]) for i in range(max(0, len(tokens) - 3)))
    phrase_media, media_terms, media_examples = defaultdict(set), defaultdict(set), defaultdict(list)
    clip_hints = defaultdict(int)
    for dataset, name in (("train", "SIS-Motion-54K.jsonl"), ("test", "SIS-Bench.jsonl")):
        for row in read_rows(data / "raw" / name):
            media = (dataset, row.get("video_path", row.get("video_name")))
            tokens = words(positive_annotation_text(row))
            media_terms[media].update(tokens)
            for index in range(max(0, len(tokens) - 3)):
                gram = tuple(tokens[index:index + 4])
                if gram in grams_wanted:
                    phrase_media[gram].add(media)
            clip_mentions = re.findall(r"\b(first|second|third|fourth|fifth) clip\b", positive_annotation_text(row), re.I)
            if clip_mentions:
                clip_hints[media] = max(clip_hints[media], max(("first", "second", "third", "fourth", "fifth").index(v.lower()) + 1 for v in clip_mentions))
            if len(media_examples[media]) < 5:
                media_examples[media].append({"id": row.get("id", row.get("question_id")),
                                             "task": row.get("task_type"), "text": positive_annotation_text(row)[:600]})
    hint_rows = []
    for row in queue + train:
        tokens = words(anchor(row["question"]))
        hints = []
        for index in range(max(0, len(tokens) - 3)):
            gram = tuple(tokens[index:index + 4])
            matches = phrase_media.get(gram, set())
            if 2 <= len(matches) <= 30:
                hints.append({"phrase": " ".join(gram), "media_count": len(matches),
                              "media": [{"dataset": dataset, "source_path": path}
                                        for dataset, path in sorted(matches)]})
        hint_rows.append({"id": row["id"], "source_path": row["source_path"],
                          "max_clip_ordinal_mentioned_elsewhere": clip_hints[(row["dataset"], row["source_path"])],
                          "unconfirmed_text_overlap_hints": hints,
                          "warning": "Annotation similarity and ordinal mentions require media verification."})
    write_rows(output / "text_overlap_review_hints.jsonl", hint_rows)
    # Donor search is a metadata shortlist, never a visual role approval.
    protected = {("test", row["source_path"]) for row in queue + excluded}
    term_media = defaultdict(set)
    for media, terms in media_terms.items():
        if media not in protected:
            for term in terms:
                term_media[term].add(media)
    donor_rows = []
    for row in queue:
        options = []
        for label, text in row["options"].items():
            if label == row["answer"]:
                continue
            wanted = set(words(text))
            class_terms = wanted - ATTRIBUTES
            pool = set().union(*(term_media.get(term, set()) for term in class_terms)) if class_terms else set()
            ranked = []
            for media in pool:
                coverage = len(wanted & media_terms[media]) / max(1, len(wanted))
                if coverage >= 0.6:
                    ranked.append((coverage, media))
            ranked.sort(key=lambda pair: (-pair[0], pair[1][0] != "test", stable(str(pair[1]), args.seed)))
            options.append({"wrong_option": label, "landmark_description": text,
                            "matching_class": family(text), "candidates": [
                                {"dataset": media[0], "source_path": media[1], "token_coverage": score,
                                 "annotation_examples": media_examples[media]}
                                for score, media in ranked[:3]]})
        donor_rows.append({"target_id": row["id"], "search": options,
                           "role": "competing donor candidates require visual audit and source reservation",
                           "neutral_rule": "Search the same scene family for footage excluding query landmark and plausible "
                                           "alternatives where practical; absence of annotation mentions is not absence evidence."})
    write_rows(output / "donor_search_hints.jsonl", donor_rows)
    report["auxiliary_index"] = {"media_indexed": len(media_terms),
                                 "rows_with_unconfirmed_overlap_hints": sum(bool(row["unconfirmed_text_overlap_hints"]) for row in hint_rows),
                                 "test_rows_with_unconfirmed_overlap_hints": sum(row["id"].startswith("test:") and bool(row["unconfirmed_text_overlap_hints"]) for row in hint_rows),
                                 "train_rows_with_unconfirmed_overlap_hints": sum(row["id"].startswith("train:") and bool(row["unconfirmed_text_overlap_hints"]) for row in hint_rows),
                                 "donor_rows": len(donor_rows),
                                 "targets_with_any_donor_search_candidate": sum(any(option["candidates"] for option in row["search"]) for row in donor_rows),
                                 "wrong_options_with_donor_search_candidate": sum(bool(option["candidates"]) for row in donor_rows for option in row["search"]),
                                 "note": "All59,154 released annotations indexed; extracted mentions can be erroneous, negative, or "
                                         "incomplete. Shortlists require clip-level inspection and duplicate screening."}
    (output / "counts_and_rules.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
