from typing import Any
ANCHOR_PREFIX = "[NOR][ANO]"

def parse_generated(text: str, active_classes: list[str], seg_token: str = "[SEG]") -> dict[str, Any]:
    found = []
    entries = []
    errors = []
    terminated = False
    none_seen = False
    seg_index = 0
    lines = text.splitlines()
    if lines and lines[0].strip() == ANCHOR_PREFIX:
        lines = lines[1:]
    elif any(ANCHOR_PREFIX in line for line in lines):
        errors.append("misplaced_anchor_prefix")
    for line in lines:
        line = line.strip()
        if line == "END":
            terminated = True
            break
        if line == "NONE":
            none_seen = True
            continue
        if not line:
            continue
        if not line.endswith(seg_token):
            errors.append("missing_SEG:" + line)
            continue
        name = line[:-len(seg_token)].strip()
        if name not in active_classes:
            errors.append("unknown_class:" + name)
        else:
            found.append(name)
            entries.append({"disease": name, "seg_index": seg_index})
        seg_index += 1
    if not terminated:
        errors.append("missing_END")
    if none_seen and found:
        errors.append("NONE_with_classes")
    if not none_seen and not found:
        errors.append("empty_output")
    return {"classes": sorted(set(found), key=active_classes.index), "entries": entries,
            "seg_count": seg_index, "errors": errors,
            "status": "ok" if not errors else "partial_or_failed", "raw_text": text}
