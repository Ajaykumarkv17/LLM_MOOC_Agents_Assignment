from typing import Dict, List
from autogen import ConversableAgent, AssistantAgent
import sys
import os
import math
from dotenv import load_dotenv



load_dotenv()

def fetch_restaurant_data(restaurant_name: str) -> Dict[str, List[str]]:
    """
    Fetch reviews for a specific restaurant from the restaurant-reviews.txt file.
    
    Args:
        restaurant_name (str): Name of the restaurant to fetch reviews for
    
    Returns:
        Dict[str, List[str]]: Dictionary with restaurant name and its reviews
    """
    try:
        with open('restaurant-data.txt', 'r') as f:
            reviews = f.readlines()
    except FileNotFoundError:
        return {restaurant_name: []}
    
    matching_reviews = [
        review.strip().split('. ', 1)[1] 
        for review in reviews 
        if review.strip().startswith(restaurant_name + '.')
    ]
    
    return {restaurant_name: matching_reviews}

def calculate_overall_score(restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]) -> Dict[str, float]:
    """
    Calculate the overall restaurant score using a geometric mean approach.
    
    Args:
        restaurant_name (str): Name of the restaurant
        food_scores (List[int]): List of food quality scores (1-5)
        customer_service_scores (List[int]): List of customer service scores (1-5)
    
    Returns:
        Dict[str, float]: Dictionary with restaurant name and its overall score
    """
    if not food_scores or not customer_service_scores:
        return {restaurant_name: 0.000}
    
    N = len(food_scores)
    
    # Compute geometric mean with penalty on food quality
    total_score = sum(
        math.sqrt(food_scores[i]**2 * customer_service_scores[i]) * (1 / (N * math.sqrt(125))) * 10 
        for i in range(N)
    )
    
    return {restaurant_name: round(total_score, 3)}

def get_data_fetch_agent_prompt(restaurant_query: str) -> str:
    """
    Generate a prompt for the data fetch agent to retrieve restaurant reviews.
    
    Args:
        restaurant_query (str): Restaurant name to fetch reviews for
    
    Returns:
        str: Prompt for the data fetch agent
    """
    return f"""
    You are a data fetch agent responsible for retrieving restaurant reviews.
    Your task is to fetch reviews for the restaurant: '{restaurant_query}'.
    
    Instructions:
    1. Call the fetch_restaurant_data function with the restaurant name.
    2. Return the fetched reviews.
    3. If no reviews are found, return an empty list.
    
    Use the fetch_restaurant_data function to accomplish this task.
    """

def get_review_analyzer_prompt() -> str:
    """
    Generate a prompt for the review analyzer agent.
    
    Returns:
        str: Prompt for the review analyzer agent
    """
    return """
    You are a review analysis agent tasked with scoring restaurant reviews.
    
    Scoring Guidelines:
    Food Score Keywords:
    - 1/5: awful, horrible, disgusting
    - 2/5: bad, unpleasant, offensive
    - 3/5: average, uninspiring, forgettable
    - 4/5: good, enjoyable, satisfying
    - 5/5: awesome, incredible, amazing
    
    Instructions:
    1. Analyze each review for food and customer service keywords
    2. Assign scores based on the first matching keyword found
    3. Return lists of food and customer service scores
    
    Ensure you process all reviews systematically and objectively.
    """

def get_scoring_agent_prompt() -> str:
    """
    Generate a prompt for the scoring agent.
    
    Returns:
        str: Prompt for the scoring agent
    """
    return """
    You are a scoring agent responsible for calculating the overall restaurant score.
    
    Your task:
    1. Review the food and customer service scores
    2. Use the calculate_overall_score function to compute the final score
    3. Return the restaurant's overall rating
    
    The scoring uses a geometric mean approach that slightly penalizes food quality.
    Ensure precise calculation of the final score.
    """

def main(user_query: str):
    # LLM configuration
    llm_config = {
        "config_list": [
            {
                "model": "gpt-4o-mini", 
                "api_key": "",
                "base_url":"https://models.inference.ai.azure.com"
            }
        ]
    }
    
    # Entrypoint Agent
    entrypoint_agent = ConversableAgent(
        "entrypoint_agent",
        system_message="You are the main supervisor agent coordinating restaurant review analysis.",
        llm_config=llm_config
    )
    
    # Register functions
    entrypoint_agent.register_for_llm(
        name="fetch_restaurant_data", 
        description="Fetches the reviews for a specific restaurant."
    )(fetch_restaurant_data)
    entrypoint_agent.register_for_execution(
        name="fetch_restaurant_data"
    )(fetch_restaurant_data)
    
    # Data Fetch Agent
    data_fetch_agent = AssistantAgent(
        "data_fetch_agent",
        system_message=get_data_fetch_agent_prompt(user_query),
        llm_config=llm_config,
    )
    
    # Review Analyzer Agent
    review_analyzer_agent = AssistantAgent(
        "review_analyzer_agent",
        system_message=get_review_analyzer_prompt(),
        llm_config=llm_config
    )
    
    # Scoring Agent
    scoring_agent = AssistantAgent(
        "scoring_agent",
        system_message=get_scoring_agent_prompt(),
        llm_config=llm_config,
        
    )
    
    # Chat Sequence
    result = entrypoint_agent.initiate_chats([
        {
            "recipient": data_fetch_agent,
            "message": f"Fetch reviews for the restaurant: {user_query}",
            "max_turns": 1
        },
        {
            "recipient": review_analyzer_agent,
            "message": "Analyze the reviews for food and customer service scores",
            "max_turns": 1
        },
        {
            "recipient": scoring_agent,
            "message": "Calculate the overall restaurant score",
            "max_turns": 1
        }
    ])
    
    # Return final result
    return result

if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please ensure you include a query for some restaurant when executing main."
    main(sys.argv[1])