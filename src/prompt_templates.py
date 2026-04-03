"""
Prompt Templates — Week 2

CO-STAR framework and Week 2-specific prompt templates.
"""

from typing import Dict, Optional


class COSTARTemplate:
    """CO-STAR prompt framework template"""

    @staticmethod
    def build(
        context: str,
        objective: str,
        style: str = "professional",
        tone: str = "helpful",
        audience: str = "general",
        response_format: str = "text"
    ) -> str:
        """Build a CO-STAR prompt."""
        return f"""# Context
{context}

# Objective
{objective}

# Style
{style}

# Tone
{tone}

# Audience
{audience}

# Response Format
{response_format}
"""

    @staticmethod
    def build_system(
        style: str = "professional",
        tone: str = "helpful",
        response_format: str = "text"
    ) -> str:
        """Build a system prompt with S, T, R components."""
        return f"""You are a helpful AI assistant.

Style: {style}
Tone: {tone}
Response Format: {response_format}

Follow these guidelines in all responses."""


class PromptLibrary:
    """Library of pre-built prompts for Week 2 topics"""

    RESEARCH_ASSISTANT = COSTARTemplate.build(
        context="You are helping a researcher gather and analyze information.",
        objective="Find relevant information and synthesize it clearly.",
        style="academic but accessible",
        tone="objective and thorough",
        audience="researcher or student",
        response_format="structured with sources cited"
    )

    CODE_REVIEWER = COSTARTemplate.build(
        context="You are reviewing code for quality, bugs, and best practices.",
        objective="Identify issues and suggest improvements.",
        style="technical and precise",
        tone="constructive and educational",
        audience="software developer",
        response_format="list of findings with code examples"
    )

    ML_EXPLAINER = COSTARTemplate.build(
        context="You are an expert ML engineer explaining concepts to a student.",
        objective="Explain machine learning concepts clearly with intuition and examples.",
        style="technical but accessible",
        tone="encouraging and educational",
        audience="ML engineering student",
        response_format="explanation with analogies, then technical details"
    )

    DATA_STRATEGIST = COSTARTemplate.build(
        context="You are a data engineering expert helping design training data pipelines.",
        objective="Advise on data collection, cleaning, and quality for LLM pretraining.",
        style="practical and actionable",
        tone="direct and expert",
        audience="ML engineer or data scientist",
        response_format="structured recommendations with trade-offs"
    )

    TUTOR = COSTARTemplate.build(
        context="You are teaching a concept to a student.",
        objective="Explain clearly and verify understanding.",
        style="educational and patient",
        tone="encouraging and supportive",
        audience="student or learner",
        response_format="explanations with examples and follow-up questions"
    )

    @classmethod
    def get_template(cls, name: str) -> Optional[str]:
        """Get a template by name."""
        return getattr(cls, name.upper(), None)

    @classmethod
    def list_templates(cls) -> list:
        """List all available templates."""
        return [attr for attr in dir(cls)
                if not attr.startswith('_') and attr.isupper()]
