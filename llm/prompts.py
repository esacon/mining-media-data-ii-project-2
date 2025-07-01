from typing import List, Tuple


class PromptTemplate:
    """Base class for prompt templates."""

    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        """Format the template with provided arguments."""
        return self.template.format(**kwargs)


class ZeroShotPrompt(PromptTemplate):
    """Zero-shot prompt for critical error detection using WMT21 guidelines."""

    def __init__(self):
        template = (
            "You are an expert in machine translation quality assessment "
            "following WMT21 guidelines. Your task is to determine whether "
            "a machine translation contains CRITICAL errors.\n\n"
            "CRITICAL ERROR CATEGORIES (WMT21):\n\n"
            "1. TOXICITY (TOX): Introduces, removes, or mistranslates "
            "hate speech, violence, or profanity\n"
            "2. SAFETY RISKS (SAF): Introduces/removes content that could "
            "endanger health/safety\n"
            "3. NAMED ENTITIES (NAM): Incorrectly translates, deletes, or "
            "introduces names, places, organizations\n"
            "4. SENTIMENT POLARITY (SEN): Introduces/removes negation, "
            "reverses sentiment\n"
            "5. NUMBERS/UNITS (NUM): Incorrectly translates numbers, dates, "
            "times, measurements\n\n"
            "NOT CRITICAL (ignore these):\n"
            "- Minor grammatical mistakes\n"
            "- Stylistic differences\n"
            "- Awkward but comprehensible phrasing\n"
            "- Preserving source language toxicity (not an error)\n\n"
            "Source text: {source}\n"
            "Translation: {target}\n\n"
            "IMPORTANT: Respond with EXACTLY ONE WORD ONLY. No explanations, "
            "no reasoning, no additional text.\n\n"
            "If there are critical errors: ERR\n"
            "If there are no critical errors: NOT\n\n"
            "Your response (one word only):"
        )
        super().__init__(template)


class FewShotPrompt(PromptTemplate):
    """Few-shot prompt with WMT21 examples for critical error detection."""

    def __init__(self):
        template = (
            "You are an expert in machine translation quality assessment "
            "following WMT21 guidelines. Your task is to determine whether "
            "a machine translation contains CRITICAL errors.\n\n"
            "CRITICAL ERROR CATEGORIES (WMT21):\n"
            "1. TOXICITY (TOX): Introduces, removes, or mistranslates "
            "hate speech, violence, or profanity\n"
            "2. SAFETY RISKS (SAF): Introduces/removes content that could "
            "endanger health/safety\n"
            "3. NAMED ENTITIES (NAM): Incorrectly translates, deletes, or "
            "introduces names, places, organizations\n"
            "4. SENTIMENT POLARITY (SEN): Introduces/removes negation, "
            "reverses sentiment\n"
            "5. NUMBERS/UNITS (NUM): Incorrectly translates numbers, dates, "
            "times, measurements\n\n"
            "Examples (respond with EXACTLY ONE WORD ONLY):\n\n"
            "Source: You're a band of brothers\n"
            "Translation: You're a band of bastards\n"
            "ERR\n\n"
            "Source: Wash your hands before eating\n"
            "Translation: Shake hands before eating\n"
            "ERR\n\n"
            "Source: LG releases new smartphone\n"
            "Translation: Samsung releases new smartphone\n"
            "ERR\n\n"
            "Source: Don't buy this product!\n"
            "Translation: Do buy this product!\n"
            "ERR\n\n"
            "Source: Drive 20 kilometers to the city\n"
            "Translation: Drive 20 miles to the city\n"
            "ERR\n\n"
            "Source: The weather is nice today\n"
            "Translation: Das Wetter ist heute schön\n"
            "NOT\n\n"
            "Source: Please call me when you arrive\n"
            "Translation: Bitte rufen Sie mich an, wenn Sie ankommen\n"
            "NOT\n\n"
            "Now analyze this translation:\n\n"
            "Source text: {source}\n"
            "Translation: {target}\n\n"
            "IMPORTANT: Respond with EXACTLY ONE WORD ONLY. No explanations, "
            "no reasoning, no additional text.\n\n"
            "If there are critical errors: ERR\n"
            "If there are no critical errors: NOT\n\n"
            "Your response (one word only):"
        )
        super().__init__(template)


def get_prompt_template(prompt_type: str) -> PromptTemplate:
    """Get prompt template by type."""
    templates = {
        "zero_shot": ZeroShotPrompt(),
        "few_shot": FewShotPrompt(),
    }

    if prompt_type not in templates:
        raise ValueError(
            f"Unknown prompt type: {prompt_type}. "
            f"Available: {list(templates.keys())}"
        )

    return templates[prompt_type]


def create_few_shot_examples() -> List[Tuple[str, str, str]]:
    """Create WMT21-based few-shot examples for critical error detection.

    Returns:
        List of (source, target, label) tuples based on WMT21 categories
    """
    examples = [
        # TOXICITY errors
        ("You're a band of brothers", "You're a band of bastards", "ERR"),
        ("He's a great leader", "He's a terrible dictator", "ERR"),
        # SAFETY RISKS errors
        ("Wash your hands before eating", "Shake hands before eating", "ERR"),
        ("Take one pill twice daily", "Take two pills twice daily", "ERR"),
        # NAMED ENTITY errors
        ("LG releases new smartphone", "Samsung releases new smartphone", "ERR"),
        ("Visit Paris next summer", "Visit London next summer", "ERR"),
        # SENTIMENT POLARITY errors
        ("Don't buy this product!", "Do buy this product!", "ERR"),
        ("This movie is not good", "This movie is good", "ERR"),
        # NUMBERS/UNITS errors
        ("Drive 20 kilometers to the city", "Drive 20 miles to the city", "ERR"),
        ("The meeting is at 3 PM", "The meeting is at 3 AM", "ERR"),
        # No critical errors
        ("The weather is nice today", "Das Wetter ist heute schön", "NOT"),
        (
            "Please call me when you arrive",
            "Bitte rufen Sie mich an, wenn Sie ankommen",
            "NOT",
        ),
        ("I like coffee in the morning", "J'aime le café le matin", "NOT"),
    ]

    return examples
