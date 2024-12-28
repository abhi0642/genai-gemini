import os
import google.generativeai as genai

def get_user_input():
    topic = input("Enter the topic for discussion: ")
    loops = int(input("Enter the number of reasoning loops: "))
    return topic, loops

def run_reasoning_chain(model, initial_topic, num_loops):
    # First agent - More focused and conservative (temperature = 0.3)
    first_config = genai.GenerationConfig(temperature=0.3)
    current_insight = model.generate_content(
        f"""Given the topic: {initial_topic}, 
        formulate a set of detailed instructions as if you were adding a detailed system prompt for advanced AI agent
        That encourages it to approach the topic from many different angles.""",
        generation_config=first_config,
        stream=True
    )
    
    all_insights = []
    current_text = ""
    for chunk in current_insight:
        current_text += chunk.text
        print(chunk.text, end="")
    all_insights.append(current_text)
    
    # Second agent - More creative and explorative (temperature = 1.0)
    second_config = genai.GenerationConfig(temperature=1.0)
    for i in range(num_loops):
        print(f"\n=== Reasoning Loop {i+1} ===\n")
        
        prompt = f"""Based on this insight: '{current_text}', 
        Please analyze this with the depth and rigor of a Nobel laureate. Consider:
        - The fundamental principles and assumptions underlying this insight
        - Novel theoretical frameworks that could reframe our understanding
        - Potential paradigm shifts this might suggest
        - Cross-disciplinary implications and connections
        - Critical questions that challenge conventional wisdom
        
        Approach this with the innovative thinking that pushes boundaries of human knowledge."""
        
        response = model.generate_content(prompt, generation_config=second_config, stream=True)
        
        current_text = ""
        for chunk in response:
            current_text += chunk.text
            print(chunk.text, end="")
        all_insights.append(current_text)
    
    # Third agent - More precise and deterministic (temperature = 0.1)
    print("\n\n=== Final Summary ===\n")
    third_config = genai.GenerationConfig(temperature=0.1)
    summary_prompt = f"""As a summarizing agent, review all the insights from our discussion on '{initial_topic}':

{chr(10).join(f'Loop {i+1}: {insight}' for i, insight in enumerate(all_insights))}

Please provide:
1. Key themes and patterns that emerged
2. Most significant insights discovered
3. A concise synthesis of the entire discussion"""
    
    summary_response = model.generate_content(summary_prompt, generation_config=third_config, stream=True)
    for chunk in summary_response:
        print(chunk.text, end="")

def main():
    genai.configure(api_key="#")
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    topic, num_loops = get_user_input()
    run_reasoning_chain(model, topic, num_loops)

if __name__ == "__main__":
    main()
