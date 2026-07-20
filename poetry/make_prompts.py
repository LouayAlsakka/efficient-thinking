#!/usr/bin/env python
"""ET-IV E1 — build the form-validity prompt set (deterministic, no model).

Three tasks, 100 prompts each, over a fixed topic list:
  - sonnet   : one quatrain, ABAB, iambic pentameter
  - villanelle: one tercet, ABA, iambic pentameter
  - lyric    : one line matching a named syllable-stress template (3 templates, cycled)
Output: poetry/data/e1_prompts.jsonl — one JSON object per prompt with the fields the checker needs
(task, scheme, target meter/template). Generation (poetry_gen.py) and scoring (e1_score.py) consume it.
"""
import json, os

TOPICS = [
    "the sea at dawn", "a fading autumn", "lost love", "the passage of time", "a winter night",
    "spring rain", "an old friend", "the moon", "a distant city", "childhood memory",
    "a burning candle", "the mountain wind", "a quiet garden", "the first snow", "a river's journey",
    "solitude", "a summer storm", "the harvest", "a sleeping child", "the northern star",
    "a ruined tower", "morning mist", "the tide", "a nightingale", "falling leaves",
    "the desert", "a lighthouse", "twilight", "an empty road", "the orchard in bloom",
    "a grandmother's hands", "the last train", "a field of wheat", "the ocean floor", "a candle's smoke",
    "the migrating geese", "a frozen lake", "the city at night", "an abandoned house", "the first love",
    "a soldier's return", "the potter's wheel", "a spider's web", "the eclipse", "a mother's song",
    "the blacksmith", "a paper boat", "the church bell", "a wild horse", "the vineyard",
    "a broken clock", "the tundra", "a street musician", "the comet", "a forgotten letter",
    "the coral reef", "a widow's grief", "the wheat mill", "a caged bird", "the glacier",
    "a fisherman's net", "the meadow", "a dying fire", "the archer", "a snowdrift",
    "the weaver", "a fallen empire", "the estuary", "a child's kite", "the observatory",
    "a monastery", "the salt marsh", "a violin", "the quarry", "a wanderer",
    "the cathedral", "a summer fair", "the shipwreck", "a beekeeper", "the aurora",
    "a stone bridge", "the almond tree", "a clockmaker", "the reservoir", "a shepherd",
    "the printing press", "a moth at the window", "the canyon", "a sailor's wife", "the greenhouse",
    "a chess game", "the lighthouse keeper", "a field of poppies", "the telescope", "a train station",
    "the olive grove", "a winter fox", "the harbor", "a candle in a window", "the last leaf",
]

LYRIC_TEMPLATES = [
    {"name": "iambic octosyllable", "template": [0, 1, 0, 1, 0, 1, 0, 1],
     "desc": "8 syllables alternating unstressed-STRESSED, beginning unstressed (da-DUM da-DUM da-DUM da-DUM)"},
    {"name": "trochaic octosyllable", "template": [1, 0, 1, 0, 1, 0, 1, 0],
     "desc": "8 syllables alternating STRESSED-unstressed, beginning stressed (DUM-da DUM-da DUM-da DUM-da)"},
    {"name": "iambic hexasyllable", "template": [0, 1, 0, 1, 0, 1],
     "desc": "6 syllables alternating unstressed-STRESSED (da-DUM da-DUM da-DUM)"},
]


def main():
    os.makedirs("poetry/data", exist_ok=True)
    rows = []
    for i, topic in enumerate(TOPICS):
        rows.append({"id": f"sonnet_{i:03d}", "task": "sonnet", "topic": topic, "scheme": "ABAB",
                     "meter": "iambic_pentameter", "n_lines": 4,
                     "prompt": f"Write a single four-line stanza (a quatrain) about {topic}. "
                               f"Use strict iambic pentameter (ten syllables per line) and an ABAB rhyme "
                               f"scheme. Output only the four lines, nothing else."})
        rows.append({"id": f"villanelle_{i:03d}", "task": "villanelle", "topic": topic, "scheme": "ABA",
                     "meter": "iambic_pentameter", "n_lines": 3,
                     "prompt": f"Write a single three-line stanza (a tercet) about {topic}. "
                               f"Use iambic pentameter (ten syllables per line) and an ABA rhyme scheme. "
                               f"Output only the three lines, nothing else."})
        tpl = LYRIC_TEMPLATES[i % len(LYRIC_TEMPLATES)]
        rows.append({"id": f"lyric_{i:03d}", "task": "lyric", "topic": topic,
                     "template": tpl["template"], "template_name": tpl["name"], "n_lines": 1,
                     "prompt": f"Write a single line of song lyric about {topic} with {tpl['desc']}. "
                               f"Output only the one line, nothing else."})
    with open("poetry/data/e1_prompts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[E1] wrote poetry/data/e1_prompts.jsonl: {len(rows)} prompts "
          f"({sum(r['task']=='sonnet' for r in rows)} sonnet / "
          f"{sum(r['task']=='villanelle' for r in rows)} villanelle / "
          f"{sum(r['task']=='lyric' for r in rows)} lyric)")


if __name__ == "__main__":
    main()
