#!/usr/bin/env python
"""ET-IV E4 — build the counterpoint prompt set (deterministic, no model).

Ten public-domain cantus firmi (Fux-style diatonic exercises), each diatonically transposed to five
positions → 50 prompts (n = 50 per cell). Each prompt asks for a first-species counter-voice above the
cantus. Output: music/data/e4_prompts.jsonl with the cantus note list the scorer (checker 1c) needs.
"""
import json, os
from music21 import note

# 10 short diatonic cantus firmi (note-name lists), classic first-species subjects
CANTUS = [
    ["D4", "F4", "E4", "D4", "G4", "F4", "A4", "G4", "F4", "E4", "D4"],   # Fux Dorian
    ["C4", "D4", "F4", "E4", "F4", "G4", "E4", "D4", "C4"],
    ["C4", "E4", "D4", "F4", "E4", "G4", "F4", "E4", "D4", "C4"],
    ["G4", "A4", "B4", "G4", "C5", "B4", "A4", "G4"],
    ["E4", "G4", "F4", "E4", "A4", "G4", "F4", "E4"],
    ["F4", "G4", "A4", "F4", "D4", "E4", "F4"],
    ["A4", "G4", "F4", "E4", "D4", "F4", "E4", "D4"],
    ["C4", "D4", "E4", "F4", "G4", "F4", "E4", "D4", "C4"],
    ["D4", "E4", "F4", "G4", "A4", "G4", "F4", "E4", "D4"],
    ["G4", "F4", "E4", "D4", "C4", "E4", "D4", "C4"],
]
TRANSPOSITIONS = ["P1", "M2", "M-2", "P4", "P-4"]                          # diatonic, keep it singable


def transpose(seq, dia):
    return [note.Note(n).transpose(dia).nameWithOctave for n in seq]


def main():
    os.makedirs("music/data", exist_ok=True)
    rows = []
    k = 0
    for ci, cf0 in enumerate(CANTUS):
        for ti, dia in enumerate(TRANSPOSITIONS):
            cf = transpose(cf0, dia)
            prompt = ("Here is a cantus firmus, one note per bar: " + " ".join(cf) + ". "
                      "Write a first-species counterpoint ABOVE it: exactly one note over each cantus "
                      "note (" + str(len(cf)) + " notes total), every interval consonant (unison, third, "
                      "fifth, sixth, or octave), no parallel fifths or octaves, no voice crossing, ending "
                      "on an octave. Output ONLY the counter-voice as space-separated note names with "
                      "octaves, for example 'E5 F5 G5'. Nothing else.")
            rows.append({"id": f"cf{ci:02d}_t{ti}", "task": "counterpoint", "cantus": cf,
                         "n_notes": len(cf), "prompt": prompt, "n_lines": 1})
            k += 1
    with open("music/data/e4_prompts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[E4] wrote music/data/e4_prompts.jsonl: {len(rows)} prompts "
          f"({len(CANTUS)} cantus firmi x {len(TRANSPOSITIONS)} transpositions)")


if __name__ == "__main__":
    main()
