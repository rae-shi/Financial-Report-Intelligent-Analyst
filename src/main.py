from graph import run_agent

if __name__ == "__main__":
    # Ask the user to input the query
    query = input(f"please input something that you want to know from the nvidia report:\n")
    
    print("--- ANALYST STARTING WORK ---")
    result = run_agent(query)
    print("\nFINAL REPORT:\n", result)