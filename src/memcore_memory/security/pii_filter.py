"""
PII Detection & Redaction - excellent version
- Regex + heuristics for email, phone, SSN, credit card, API keys
- GDPR compliance: detect before storage
- Configurable action: block, redact, warn
"""
import re
from typing import Dict, List, Tuple

PII_PATTERNS = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
    # US numbers, plus an E.164-style alternative anchored on a leading "+" so
    # international numbers (+36 30 123 4567) are caught without matching dates
    # or version strings.
    "phone": re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
                        r'|\+\d{1,3}[ -]?\d{1,4}(?:[ -]?\d{2,4}){2,4}\b'),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    # A run of digit groups separated by single spaces or dashes. Which part of the
    # run is a card is decided by _card_spans below, not by the regex: a greedy
    # 13-19 digit match swallowed any digits in front ("qty 2 4111 1111 ..."),
    # failed Luhn as a whole and was never retried, so the card got through.
    "credit_card": re.compile(r'(?<!\w)\d+(?:[ -]\d+)*(?!\w)'),
    # IGNORECASE as a compile flag: an inline (?i) mid-pattern is an error on 3.11+.
    "api_key": re.compile(
        r'\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}'
        r'|\bAKIA[0-9A-Z]{16}\b'
        r'|\bgh[pousr]_[A-Za-z0-9]{36,}\b'
        r'|\bapi[_-]?key\s*[:=]\s*["\']?[A-Za-z0-9_\-./+]{16,}'
        r'|\bapi[_-]?key[_-]?[A-Za-z0-9]{20,}\b',
        re.IGNORECASE),
    "ip": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
}

# Longest, most specific kinds first: a card number redacted before the phone
# pattern runs can't be half-swallowed by it.
REDACT_ORDER = ["api_key", "credit_card", "ssn", "email", "phone", "ip"]

# How many examples of each kind scan() reports. A report cap only - redaction
# works on the whole text, not on this sample.
REPORT_LIMIT = 5


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _is_card(candidate: str) -> bool:
    digits = re.sub(r'\D', '', candidate)
    return 13 <= len(digits) <= 19 and _luhn(digits)


_GROUP = re.compile(r'\d+')


def _card_spans(text: str) -> List[Tuple[int, int]]:
    """(start, end) of every card number in text.

    Each run of digit groups is searched for windows of whole groups holding
    13-19 digits that pass Luhn, leftmost first and longest first, so a card
    is found whatever digits stand before or after it. Windows never split a
    group: a card is written as whole groups, and allowing a split would make
    about one in ten 13+ digit substrings of any long number look like a card.
    """
    spans = []
    for run in PII_PATTERNS["credit_card"].finditer(text):
        groups = [(g.start() + run.start(), g.end() + run.start(), len(g.group(0)))
                  for g in _GROUP.finditer(run.group(0))]
        i = 0
        while i < len(groups):
            found = None
            digits = 0
            for j in range(i, len(groups)):
                digits += groups[j][2]
                if digits > 19:
                    break
                if digits >= 13 and _is_card(text[groups[i][0]:groups[j][1]]):
                    found = j
            if found is None:
                i += 1
            else:
                spans.append((groups[i][0], groups[found][1]))
                i = found + 1
    return spans


def _matches(pii_type: str, text: str) -> List[str]:
    if pii_type == "credit_card":
        return [text[a:b] for a, b in _card_spans(text)]
    return [m.group(0) for m in PII_PATTERNS[pii_type].finditer(text)]


def _replace_spans(text: str, spans: List[Tuple[int, int]], marker: str) -> str:
    out, last = [], 0
    for a, b in spans:
        out.append(text[last:a])
        out.append(marker)
        last = b
    out.append(text[last:])
    return "".join(out)


class PIIFilter:
    def __init__(self, enabled: bool = True, action: str = "warn"):
        # action: warn, redact, block
        self.enabled = enabled
        self.action = action

    def scan(self, text: str) -> Dict[str, List[str]]:
        if not self.enabled:
            return {}
        findings = {}
        for pii_type in PII_PATTERNS:
            matches = _matches(pii_type, text)
            if matches:
                findings[pii_type] = matches[:REPORT_LIMIT]
        return findings

    def redact(self, text: str) -> Tuple[str, Dict]:
        # Substitute over the whole text. Replacing the reported strings one by one
        # left every match past the fifth in place, and replaced "a@b.co" inside
        # "aa@b.co" first, leaking the leftover "a".
        findings = self.scan(text)
        redacted = text
        for pii_type in REDACT_ORDER:
            marker = f"[{pii_type.upper()}_REDACTED]"
            if pii_type == "credit_card":
                redacted = _replace_spans(redacted, _card_spans(redacted), marker)
            else:
                redacted = PII_PATTERNS[pii_type].sub(marker, redacted)
        return redacted, findings

    def check_and_act(self, text: str) -> Tuple[bool, str, Dict]:
        # Returns (allowed, processed_text, findings)
        findings = self.scan(text)
        if not findings:
            return True, text, {}

        if self.action == "block" and any(k in findings for k in ["ssn", "credit_card", "api_key"]):
            return False, "", findings
        elif self.action == "redact":
            redacted, _ = self.redact(text)
            return True, redacted, findings
        else:  # warn
            return True, text, findings
