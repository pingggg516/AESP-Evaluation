"""Conservative checks on complete binary responses, without images or GT.

This is a closed grammar, not a general language parser. Unrecognized wording,
corrections, quotations, conditions and any unexplained trailing prose defer to
the frozen extractor. Coordinate-free formatting is never object-count evidence.
"""
from __future__ import annotations

import re
from typing import Any, Mapping


_FLAGS = re.I | re.ASCII
_LOCATION = r"(?: in (?:this|the) (?:aerial )?image)?"
_RELATIONS = frozenset((-1, 0, 1))


def _space(text: str) -> str:
    return " ".join(text.casefold().split())


def _category(text: str) -> str:
    return _space(re.sub(r"[-_]+", " ", text))


def _aliases(name: str) -> set[str]:
    """Only the exact category and its regular plural; no semantic synonyms."""
    name = _category(name)
    if not re.fullmatch(r"[a-z]+(?: [a-z]+)*", name, re.ASCII):
        return set()
    stem, _, word = name.rpartition(" ")
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        plural = word[:-1] + "ies"
    elif word.endswith(("s", "x", "z", "ch", "sh")):
        plural = word + "es"
    else:
        plural = word + "s"
    return {name, (stem + " " + plural).strip()}


def _category_pattern(names: set[str]) -> str:
    return "(?:" + "|".join(
        r"[ _-]+".join(re.escape(word) for word in name.split())
        for name in sorted(names, key=lambda value: (-len(value), value))
    ) + ")"


def _answer_body(text: str, names: set[str], which: dict[str, int]):
    """Parse the whole tail; derive zero counts only from explicit language.

    Empty arrays and difficulty drawing boxes make no count assertion. A closed
    'No CLASS visible in the image' assertion does. Unknown prose defers to the
    LLM instead of being discarded when selecting the main answer sentence.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return None
    if len(lines) > 1:
        labels = _category_pattern(names) if names else r'(?!)'
        neutral = (r'(?:numeric )?bounding boxes\s*:|'
                   r'no (?:bounding )?boxes(?: (?:provided|found))?[.!]?|'
                   r'no bounding boxes for the relevant objects[.!]?|'
                   r'(?:[-*#]\s*)?' + labels + r'\s*:\s*no bounding boxes[.!]?|'
                   r'(?:[-*#]\s*)?' + labels + r'(?:\s+\d+)?\s*:?[.!]?')
        lines = [line for i, line in enumerate(lines)
                 if i == 0 or not re.fullmatch(neutral, line, _FLAGS)]
    if re.fullmatch(r'(?:yes|no)', lines[0], _FLAGS):
        lines[0] += '.'
    absent = set()
    headers = [i for i, line in enumerate(lines)
               if re.match(r"boxes\s*:", line, _FLAGS)]
    if headers:
        if len(headers) != 1 or headers[0] == 0:
            return None
        start = headers[0]
        if not re.fullmatch(r"boxes\s*:\s*(?:\[\s*\])?", lines[start], _FLAGS):
            return None
        label = _category_pattern(names) if names else r"(?!)"
        tail = r"(?:\[\s*\]|(?:[-*]\s*)?" + label + r"\s*:\s*(?:\[\s*\])?)"
        annotated = r"(?:[-*]\s*)?(?P<label>" + label + r")\s*:\s*\[\s*\]\s*\((?P<note>.+)\)"
        difficulty = r"no bounding boxes for (?P<name>" + label + r") as they are not distinctly visible[.!]?"
        absence = r"no (?P<name>" + label + r") visible in (?:the|this) (?:aerial )?image[.!]?"
        for line in lines[start + 1:]:
            if re.fullmatch(tail, line, _FLAGS):
                continue
            row = re.fullmatch(annotated, _space(line), _FLAGS)
            if row is None:
                return None
            note = re.fullmatch(difficulty, row["note"], _FLAGS)
            zero = re.fullmatch(absence, row["note"], _FLAGS)
            understood = note or zero
            if understood is None or which[_category(row["label"])] != which[_category(understood["name"])]:
                return None
            if zero:
                absent.add(which[_category(zero["name"])])
        lines = lines[:start]
    body = _space(" ".join(lines))
    return re.sub(r"^answer\s*:\s*", "", body, count=1), absent


def _comparison(text: str, noun: str, question: bool, reference=None):
    if question:
        patterns = (
            r"are there (?P<op>more|fewer|less) (?P<a>" + noun + r") than (?P<b>" + noun + r")" + _LOCATION + r"\?",
            r"are there (?P<op>equal numbers of|the same number of) (?P<a>" + noun + r") and (?P<b>" + noun + r")" + _LOCATION + r"\?",
        )
    else:
        patterns = (
            r"there are (?P<neg>not )?(?P<op>more|fewer|less) (?P<a>" + noun + r") than (?P<b>" + noun + r")" + _LOCATION + r"[.!]?",
            r"there (?:is|are) (?P<neg>not )?(?P<op>equal numbers of|the same number of) (?P<a>" + noun + r") and (?P<b>" + noun + r")" + _LOCATION + r"[.!]?",
            r"(?P<a>" + noun + r") (?P<op>outnumbers?) (?P<b>" + noun + r")" + _LOCATION + r"[.!]?",
            r"(?P<a>" + noun + r") (?P<neg>(?:does|do) not )(?P<op>outnumber) (?P<b>" + noun + r")" + _LOCATION + r"[.!]?",
            r"the (?P<a>" + noun + r") and (?P<b>" + noun + r") totals are (?P<op>equal|different)" + _LOCATION + r"[.!]?",
            r"the total number of (?P<a>" + noun + r") is (?P<neg>not )?(?P<op>less|more) than (?:the total number of )?(?P<b>" + noun + r")" + _LOCATION + r"[.!]?",
            r"the (?:count|number) of (?P<a>" + noun + r") is (?P<neg>not )?(?P<op>less|more|greater) than (?:the (?:count|number) of )?(?P<b>" + noun + r")" + _LOCATION + r"[.!]?",
        )
    for pattern in patterns:
        found = re.fullmatch(pattern, text, _FLAGS)
        if found:
            value = found.groupdict()
            relation = 1 if value["op"] in ("more", "greater", "outnumber", "outnumbers") else -1 if value["op"] in ("fewer", "less") else 0
            allowed = {-1, 1} if value["op"] == "different" else {relation}
            if value.get("neg"):
                allowed = set(_RELATIONS) - allowed
            return _category(value["a"]), _category(value["b"]), allowed
    if not question and reference is not None:
        # The complete sentence names no competing referent: 'the totals'
        # denotes the two totals in the fully parsed comparison question.
        found = re.fullmatch(r"the totals are (equal|different)" + _LOCATION + r"[.!]?", text, _FLAGS)
        if found:
            return reference[0], reference[1], {0} if found[1] == "equal" else {-1, 1}
    return None


_NUMBER_WORDS = dict(zip(('zero one two three four five six seven eight nine ten '
                         'eleven twelve thirteen fourteen fifteen sixteen seventeen '
                         'eighteen nineteen twenty').split(), range(21)))
_COUNT = r'(?:[0-9]+|' + '|'.join(_NUMBER_WORDS) + r')'


def _count_claims(text, noun, which):
    """Read explicit exact counts, never item numbers, boxes or vague quantities."""
    suffix = r'(?: (?:visible|present))?' + _LOCATION
    pair = re.fullmatch(
        r'there (?:are|is) (?:(?:only|exactly|just) )?(?P<n>'+_COUNT+r') (?P<a>'+noun+r')'
        r' and (?:(?:only|exactly|just) )?(?P<m>'+_COUNT+r') (?P<b>'+noun+r')'+suffix, text, _FLAGS)
    if pair:
        if which[_category(pair['a'])] == which[_category(pair['b'])]:
            return None
        number = lambda value: int(value) if value.isdigit() else _NUMBER_WORDS[value]
        return {which[_category(pair['a'])]: number(pair['n']),
                which[_category(pair['b'])]: number(pair['m'])}
    absent = re.fullmatch(r'there (?:are|is) no (?P<a>'+noun+r')'
                          r'(?: or (?P<b>'+noun+r'))?'+suffix, text, _FLAGS)
    if absent:
        return {which[_category(absent[k])]: 0 for k in ('a', 'b') if absent[k]}
    return None


def _claims(text, noun, which, left, right):
    """Intersect complete understood assertions; any unknown clause defers."""
    allowed, counts, contradictory_counts = set(_RELATIONS), {}, False
    parts = re.split(r'[.!;](?:\s+|$)|,\s*(?:so|therefore)\s+', text)
    for part in (p.strip() for p in parts if p.strip()):
        part = re.sub(r'^therefore,?\s+', '', part)
        count = _count_claims(part, noun, which)
        if count is not None:
            for key, value in count.items():
                if key in counts and counts[key] != value:
                    contradictory_counts = True
                counts[key] = value
            continue
        claim = _comparison(part, noun, question=False, reference=(left, right))
        if claim is None:
            return None
        a, b, relation = claim
        if which[a] == which[b]:
            return None
        allowed &= relation if which[a] == which[left] else {-x for x in relation}
    if contradictory_counts:
        return set()
    a, b = which[left], which[right]
    if a in counts and b in counts:
        allowed &= {(counts[a] > counts[b]) - (counts[a] < counts[b])}
    elif counts.get(a) == 0:
        allowed &= {-1, 0}
    elif counts.get(b) == 0:
        allowed &= {0, 1}
    return allowed


def binary_guard(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a definite whole-response decision, or None to defer to the LLM.

    Only question, task_type, target_classes and answer_text are accessed. A bare
    explicit answer is safe; a comparison sentence must name precisely the two
    queried categories. All understood claims are intersected: an empty set is
    a contradiction, never a reason to privilege the first or last sentence.
    """
    task = record.get("task_type")
    text, question, classes = (record.get(key) for key in
                               ("answer_text", "question", "target_classes"))
    if (task not in ("existence", "comparison") or not isinstance(text, str)
            or not isinstance(question, str) or not question.strip()
            or not isinstance(classes, list) or not classes
            or any(not isinstance(name, str) or not name.strip() for name in classes)):
        return None
    aliases = [_aliases(name) for name in classes]
    names = set().union(*aliases)
    which = {name: index for index, group in enumerate(aliases) for name in group}
    if sum(map(len, aliases)) != len(which):
        return None
    parsed_body = _answer_body(text, names, which)
    if parsed_body is None:
        return None
    body, absent = parsed_body
    if not body:
        return None
    bare = re.fullmatch(r"(yes|no)[.!]?", body, _FLAGS)
    if bare and not absent:
        return {"status": "extracted", "answer": bare[1].lower()}
    if task != "comparison" or len(aliases) != 2 or not all(aliases) or aliases[0] & aliases[1]:
        return None
    noun = _category_pattern(names)
    query = _comparison(_space(question), noun, question=True)
    if query is None:
        return None
    left, right, query_relations = query
    if which[left] == which[right]:
        return None

    # Do not search for a yes/no token inside arbitrary language. The optional
    # leading answer and the remaining assertion must account for the full body.
    header = re.match(r"^(yes|no)[,.;!]\s+", body, _FLAGS)
    if bare:
        allowed = set(_RELATIONS)
    else:
        statement = body[header.end():] if header else body
        if header:
            statement = re.sub(r"^because ", "", statement, count=1)
        allowed = _claims(statement, noun, which, left, right)
        if allowed is None:
            return None
    explicit = bare or header
    if explicit:
        stated_answer = query_relations if explicit[1].lower() == "yes" else set(_RELATIONS) - query_relations
        allowed &= stated_answer
    # Nonnegative object counts: left=0 implies left<=right; right=0 implies
    # left>=right. Both absent imply equality. These are linguistic assertions,
    # never deductions from the number or coordinates of returned boxes.
    if which[left] in absent:
        allowed &= {-1, 0}
    if which[right] in absent:
        allowed &= {0, 1}
    if not allowed:
        return {"status": "unresolved", "answer": None}
    if allowed <= query_relations:
        return {"status": "extracted", "answer": "yes"}
    if not (allowed & query_relations):
        return {"status": "extracted", "answer": "no"}
    return None
