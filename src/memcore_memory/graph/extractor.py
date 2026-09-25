
import re

# A word of unicode letters/digits starting with a letter, hyphenated parts
# allowed. The old pattern was ASCII-only and wanted Capital+lowercase, so it
# could not produce "SkyNAS", "WireGuard" or "Kovács-Dobos Ádám" - this
# project's own names - and turned the last into "Dobos".
_WORD = re.compile(r'[^\W\d_][^\W_]*(?:-[^\W\d_][^\W_]*)*')
_STOP = {'the', 'this', 'that', 'these', 'those', 'i', 'we', 'you', 'he', 'she', 'it', 'they',
         'a', 'an', 'and', 'or', 'but', 'if', 'in', 'on', 'at', 'to', 'of', 'for', 'my', 'our'}


def extract_entities(text: str, limit: int = 10) -> list[str]:
    """Capitalised words and runs of them, as candidate entity names.

    A heuristic, not NER: "Home Assistant" comes back as one name, and a
    capitalised sentence opener that is not a stop word is a false positive.
    Deduplicated case-insensitively, since graph node ids ignore case.

    MnemosyneMemory.add calls this only when MNEM_AUTO_EXTRACT_ENTITIES is set
    and the caller passed no entities. Off by default: extracting on every write
    would change graph ranking for the existing stores, which were indexed from
    caller-supplied entities alone.
    """
    out, seen = [], set()
    run = []
    last_end = None

    def flush():
        if run:
            name = ' '.join(run)
            if name.lower() not in seen:
                seen.add(name.lower())
                out.append(name)
            run.clear()

    for m in _WORD.finditer(text or ''):
        word = m.group(0)
        capital = word[0].isupper() and word.lower() not in _STOP
        # Only plain spaces join a run; punctuation between words ends it.
        adjacent = last_end is not None and text[last_end:m.start()].strip(' ') == '' and text[last_end:m.start()] != ''
        if capital and run and adjacent:
            run.append(word)
        else:
            flush()
            if capital:
                run.append(word)
        last_end = m.end()
    flush()
    return out[:limit]
