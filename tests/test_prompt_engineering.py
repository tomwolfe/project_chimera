import pytest
from src.utils.prompt_engineering import create_persona_prompt, create_task_prompt

def test_create_persona_prompt_basic():
    """Test creating a persona prompt with basic details."""
    persona_details = {
        "name": "Test Persona",
        "role": "Tester",
        "goal": "Evaluate prompts"
    }
    expected_prompt = "You are Test Persona, a Tester. Your goal is to Evaluate prompts."
    assert create_persona_prompt(persona_details) == expected_prompt

def test_create_persona_prompt_with_constraints():
    """Test creating a persona prompt with additional constraints."""
    persona_details = {
        "name": "Constraint Bot",
        "role": "Rule Enforcer",
        "goal": "Ensure adherence to rules",
        "constraints": ["Be concise", "Avoid jargon"]
    }
    expected_prompt = "You are Constraint Bot, a Rule Enforcer. Your goal is to Ensure adherence to rules. Adhere to the following constraints: Be concise, Avoid jargon."
    assert create_persona_prompt(persona_details) == expected_prompt

def test_create_persona_prompt_empty_details():
    """Test creating a persona prompt with empty details."""
    persona_details = {}
    expected_prompt = "You are an AI assistant. Your goal is to assist the user."
    assert create_persona_prompt(persona_details) == expected_prompt

def test_create_task_prompt_basic():
    """Test creating a basic task prompt."""
    task_description = "Summarize the provided text."
    expected_prompt = f"Task: {task_description}\n\nProvide a concise summary."
    assert create_task_prompt(task_description) == expected_prompt

def test_create_task_prompt_with_context():
    """Test creating a task prompt with context."""
    task_description = "Analyze the user query."
    context = "User is asking about project status."
    expected_prompt = f"Task: {task_description}\n\nContext: {context}\n\nProvide a detailed analysis."
    assert create_task_prompt(task_description, context=context) == expected_prompt

def test_create_task_prompt_with_specific_instructions():
    """Test creating a task prompt with specific output instructions."""
    task_description = "Extract key entities."
    instructions = "Output the entities as a JSON list."
    expected_prompt = f"Task: {task_description}\n\nInstructions: {instructions}\n\nProvide the extracted entities in the specified format."
    assert create_task_prompt(task_description, instructions=instructions) == expected_prompt

# Add more tests for edge cases and variations in input
