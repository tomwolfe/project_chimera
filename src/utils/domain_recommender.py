# src/utils/domain_recommender.py
from typing import Dict, Any, Optional, List # Import List

def recommend_domain_from_keywords(user_prompt: str, domain_keywords: Dict[str, List[str]]) -> Optional[str]:
    """
    Recommends a domain/framework based on keywords in the user prompt.
    
    Args:
        user_prompt: The user's input prompt
        domain_keywords: A dictionary mapping domain names to lists of keywords
        
    Returns:
        The recommended domain name, or None if no strong match is found
    """
    if not user_prompt or not domain_keywords:
        return None
        
    # Convert prompt to lowercase for case-insensitive matching
    prompt_lower = user_prompt.lower()
    
    best_match = None
    highest_score = 0
    
    # Define a default weight for each keyword if not specified in config.yaml
    # Based on config.yaml, keywords are just strings in a list, not dicts with weights.
    DEFAULT_KEYWORD_WEIGHT = 1.0 
    
    for domain, keywords_list in domain_keywords.items(): # 'keywords_list' is now a list of strings
        # Ensure keywords_list is actually a list
        if not isinstance(keywords_list, list):
            continue # Skip if format is unexpected
            
        score = 0
        for keyword in keywords_list: # Iterate directly over the list of keywords
            keyword_lower = keyword.lower()
            # Check if the keyword exists in the prompt
            if keyword_lower in prompt_lower:
                score += DEFAULT_KEYWORD_WEIGHT # Add default weight
                
        # If we found matching keywords and the score is higher than the current best
        if score > 0 and score > highest_score:
            highest_score = score
            best_match = domain
            
    # Only return a recommendation if we have a reasonably strong match (score >= 1.0)
    # The threshold might need adjustment depending on how many keywords are expected to match.
    # If a single keyword match is enough for a recommendation, 1.0 is a good start.
    if highest_score >= 1.0:
        return best_match
        
    return None