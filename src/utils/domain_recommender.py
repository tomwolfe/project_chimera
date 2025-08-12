# src/utils/domain_recommender.py
from typing import Dict, Any, Optional

def recommend_domain_from_keywords(user_prompt: str, domain_keywords: Dict[str, Any]) -> Optional[str]:
    """
    Recommends a domain/framework based on keywords in the user prompt.
    
    Args:
        user_prompt: The user's input prompt
        domain_keywords: A dictionary mapping domain names to their keyword configurations
        
    Returns:
        The recommended domain name, or None if no strong match is found
    """
    if not user_prompt or not domain_keywords:
        return None
        
    # Convert prompt to lowercase for case-insensitive matching
    prompt_lower = user_prompt.lower()
    
    best_match = None
    highest_score = 0
    
    for domain, config in domain_keywords.items():
        # Skip if no keywords defined for this domain
        if "keywords" not in config:
            continue
            
        score = 0
        # Iterate through keywords and their weights for the current domain
        for keyword, weight in config["keywords"].items():
            keyword_lower = keyword.lower()
            # Check if the keyword exists in the prompt
            if keyword_lower in prompt_lower:
                score += weight
                
        # If we found matching keywords and the score is higher than the current best
        if score > 0 and score > highest_score:
            highest_score = score
            best_match = domain
            
    # Only return a recommendation if we have a reasonably strong match (score >= 1.0)
    if highest_score >= 1.0:
        return best_match
        
    return None
