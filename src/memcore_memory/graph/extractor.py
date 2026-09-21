
import re
def extract_entities(text: str) -> list[str]:
    pattern = r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b'
    candidates = re.findall(pattern, text)
    stop = {'The','This','That','I','We','You'}
    return [c for c in candidates if c not in stop][:10]
