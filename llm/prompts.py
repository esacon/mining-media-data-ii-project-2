from typing import List, Tuple


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
            "TASK: Detect critical errors in machine translation using WMT21 guidelines.\n"  # noqa: E501
            "CRITICAL ERROR CATEGORIES:\n"
            "1. TOXICITY: hate speech, violence, profanity added/removed\n"
            "2. SAFETY RISKS: health/safety content removed/added  \n"
            "3. NAMED ENTITIES: names, places, organizations wrong\n"
            "4. SENTIMENT POLARITY: meaning reversed due to negation\n"
            "5. NUMBERS/UNITS: incorrect numbers, dates, times, measurements\n\n"
            "IGNORE: grammar mistakes, style differences, awkward phrasing\n\n"
            "SOURCE: {source}\n"
            "TARGET: {target}\n\n"
            "DO NOT:\n"
            "- Add explanations\n"
            "- Add reasoning\n"
            "- Add examples\n"
            "- Add punctuation\n"
            "- Add anything after your answer\n\n"
            "RESPOND WITH EXACTLY ONE WORD:\n"
            "- ERR (if critical errors exist)\n"
            "- NOT (if no critical errors)\n\n"
            "ANSWER:"
        )
        super().__init__(template)


class FewShotPrompt(PromptTemplate):
    """Few-shot prompt with WMT21 examples for critical error detection."""

    examples = create_few_shot_examples()
    examples_str = "\n".join(
        [
            f"SOURCE: {source}\nTARGET: {target}\nANSWER: {label}\n"
            for source, target, label in examples
        ]
    )

    def __init__(self):
        template = (
            "TASK: Detect critical errors in machine translation using WMT21 guidelines.\n"  # noqa: E501
            "CRITICAL ERROR CATEGORIES:\n"
            "1. TOXICITY: hate speech, violence, profanity added/removed\n"
            "2. SAFETY RISKS: health/safety content removed/added  \n"
            "3. NAMED ENTITIES: names, places, organizations wrong\n"
            "4. SENTIMENT POLARITY: meaning reversed due to negation\n"
            "5. NUMBERS/UNITS: incorrect numbers, dates, times, measurements\n\n"
            "IGNORE: grammar mistakes, style differences, awkward phrasing\n\n"
            "EXAMPLES:\n"
            f"{self.examples_str}\n"
            "SOURCE: {source}\n"
            "TARGET: {target}\n\n"
            "DO NOT:\n"
            "- Add explanations to your answer\n"
            "- Add reasoning\n"
            "- Add examples\n"
            "- Add punctuation\n"
            "- Add anything after your answer\n\n"
            "RESPOND WITH EXACTLY ONE WORD:\n"
            "- ERR (if critical errors exist)\n"
            "- NOT (if no critical errors)\n\n"
            "ANSWER:"
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
