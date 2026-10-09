"""Prepare language for answer extraction; never obtain scoring boxes from text."""
import json
import math
import re


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field: '+key)
        result[key] = value
    return result


def _nonfinite_json(value):
    raise ValueError('Nonfinite JSON value: '+value)


def _json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Nonfinite JSON number: '+value)
    return number


def _json_comment_view(text, start, comments=None):
    """Mask comments inside one JSON candidate without moving source offsets.

    Quoted strings and prose outside containers stay intact. This is only for
    model response text; submitted JSONL files remain strict JSON.
    """
    chars = list(text)
    depth = 0
    quoted = escaped = False
    pos = start
    while pos < len(text):
        char = text[pos]
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in '[{':
            depth += 1
        elif char in ']}':
            depth -= 1
            if depth == 0:
                break
        elif depth and (text.startswith('//', pos) or text.startswith('/*', pos)):
            if text.startswith('//', pos):
                end = pos + 2
                while end < len(text) and text[end] not in '\r\n':
                    end += 1
            else:
                end = text.find('*/', pos + 2)
                if end < 0:
                    raise ValueError('Unclosed JSON block comment at character '+str(pos))
                end += 2
            for index in range(pos, end):
                if chars[index] not in '\r\n':
                    chars[index] = ' '
            if comments is not None:
                body = text[pos+2:end] if text.startswith('//', pos) else text[pos+2:end-2]
                comments.append((pos, body.strip()))
            pos = end
            continue
        pos += 1
    return ''.join(chars)


_JSON_DECODER = json.JSONDecoder(object_pairs_hook=_unique_pairs,
                               parse_constant=_nonfinite_json, parse_float=_json_float)


_BOX_KEYS = {'bbox', 'box', 'bounding_box', 'coordinates'}


_BOX_LIST_KEYS = {'bboxes', 'boxes', 'bounding_boxes', 'evidence_boxes'}


_FORMAT_KEYS = {'format', 'box_format', 'bbox_format', 'coordinate_format'}


def json_response_view(text):
    """Decode response objects and comments with their original positions."""
    view, objects, pos = text, [], 0
    while pos < len(view):
        if view[pos] == '"':
            try:
                _, pos = _JSON_DECODER.raw_decode(view, pos)
                continue
            except json.JSONDecodeError:
                pass
        object_start = (view[pos] == '{' and
                        re.match(r'''\{\s*(?:["']|[A-Za-z_]\w*\s*:|\}|//|/\*)''', view[pos:]))
        list_start = view[pos] == '[' and re.match(r'\[\s*(?:\{|//|/\*)', view[pos:])
        if object_start or list_start:
            comments = []
            view = _json_comment_view(view, pos, comments)
            try:
                value, end = _JSON_DECODER.raw_decode(view, pos)
            except json.JSONDecodeError as exc:
                raise ValueError('Malformed JSON object at character '+str(pos)) from exc
            objects.append({'start': pos, 'end': end, 'value': value, 'comments': comments})
            pos = end
        else:
            pos += 1
    if len(objects) > 1:
        raise ValueError('Multiple JSON objects require one unambiguous response object')
    return view, objects


def json_answer_text(text):
    """Read semantic fields in order, retaining comments and surrounding prose."""
    view, objects = json_response_view(text)
    if not objects or not isinstance(objects[0]['value'], dict):
        return None
    obj = objects[0]
    events = list(obj['comments'])
    coordinate_keys = _BOX_KEYS | _BOX_LIST_KEYS | _FORMAT_KEYS
    if {'x', 'y', 'width', 'height'} <= {key.lower() for key in obj['value']}:
        coordinate_keys = coordinate_keys | {'x', 'y', 'width', 'height'}
    pos = obj['start'] + 1
    while pos < obj['end'] - 1:
        pos = re.compile(r'\s*').match(view, pos).end()
        if view[pos] == '}':
            break
        key, end = _JSON_DECODER.raw_decode(view, pos)
        begin = re.compile(r'\s*').match(view, end).end() + 1
        begin = re.compile(r'\s*').match(view, begin).end()
        value, end = _JSON_DECODER.raw_decode(view, begin)
        if key.lower() not in coordinate_keys:
            body = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            events.append((pos, body if key.lower() == 'answer' else key + ': ' + body))
        pos = re.compile(r'\s*').match(view, end).end()
        if view[pos] == ',':
            pos += 1
    before, after = text[:obj['start']], text[obj['end']:]
    before = re.sub(r'```(?:jsonc?)?\s*$', '', before, flags=re.I)
    after = re.sub(r'^\s*```', '', after)
    return '\n'.join(part for part in [before.strip(), *(v for _, v in sorted(events)), after.strip()] if part)


def answer_view(pred):
    """Read JSON language fields when unambiguous; otherwise keep the full text.

    JSON box fields are excluded from language extraction, without parsing their
    coordinates. Explanations, corrections and comments stay in source order.
    Other response formats pass through unchanged to the frozen extractor.
    """
    text = pred['raw_answer']
    if not isinstance(text, str):
        raise ValueError('raw_answer must be a string')
    try:
        semantic = json_answer_text(text)
    except (ValueError, TypeError, IndexError):
        semantic = None
    return (semantic, 'json_semantic_fields') if semantic is not None else (text, 'original_text')
