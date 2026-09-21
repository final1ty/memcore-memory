
"""
PII Detection & Redaction - excellent version
- Regex + heuristics for email, phone, SSN, credit card, API keys
- GDPR compliance: detect before storage
- Configurable action: block, redact, warn
"""
import re
from typing import Dict, List, Tuple

PII_PATTERNS = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "phone": re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "credit_card": re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'),
    "api_key": re.compile(r'\b(?:sk-[A-Za-z0-9]{20,}|api[_-]?key[_-]?[A-Za-z0-9]{20,})\b', re.IGNORECASE),
    "ip": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
}

class PIIFilter:
    def __init__(self, enabled: bool = True, action: str = "warn"):
        # action: warn, redact, block
        self.enabled = enabled
        self.action = action

    def scan(self, text: str) -> Dict[str, List[str]]:
        if not self.enabled:
            return {}
        findings = {}
        for pii_type, pattern in PII_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                findings[pii_type] = matches[:5]  # limit
        return findings

    def redact(self, text: str) -> Tuple[str, Dict]:
        findings = self.scan(text)
        redacted = text
        for pii_type, matches in findings.items():
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                redacted = redacted.replace(match, f"[{pii_type.upper()}_REDACTED]")
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
