from scripts.train_ga import *


def create_ga_chromosome_metrics(ticker, population = 10):
    """
    Initializes a population of chromosomes using a genetic algorithm, evaluates their fitness on a dataset of technical indicators,
    and compares their performance metrics.

    Workflow:
    1. Initializes a population of chromosomes.
    2. Loads technical indicator data from a CSV file.
    3. Evaluates the fitness of each chromosome using the provided dataset.
    4. Collects and organizes the results, expanding chromosome genes into separate columns.
    5. Combines gene values with performance metrics for each chromosome.
    6. Reorders columns for clarity and comparison.
    7. Prints a formatted comparison table of chromosome performance metrics.
    8. Saves the results to a CSV file for further analysis.

    Metrics compared include:
        - Chromosome gene values (buy/sell thresholds and intervals)
        - Total return
        - Annualized return
        - Sharpe ratio
        - Sortino ratio
        - Maximum drawdown
        - Overall fitness score

    Output:
        - Prints a comparison table to the console.
        - Saves the results as 'data/GA_chromosome_metrics.csv'.
    """
    pop = initialize_population(population)
    df = pd.read_csv(f'data/{ticker}_technical_indicators.csv')

    results = []

    for i, chromo in enumerate(pop):
        metrics = evaluate_fitness(chromo, df)
        results.append(metrics)

    # Convert results to DataFrame for comparison
    results_df = pd.DataFrame(results)

    # Expand chromosome genes into separate columns
    chromo_df = pd.DataFrame(results_df['chromosome'].tolist(),
                            columns=['down_buy_val', 'down_buy_int', 'down_sell_val', 'down_sell_int',
                                    'up_buy_val', 'up_buy_int', 'up_sell_val', 'up_sell_int'])

    # Combine with metrics
    final_df = pd.concat([chromo_df, results_df.drop(columns=['chromosome'])], axis=1)

    # Reorder columns for clarity
    cols_order = ['down_buy_val', 'down_buy_int', 'down_sell_val', 'down_sell_int',
                'up_buy_val', 'up_buy_int', 'up_sell_val', 'up_sell_int',
                'total_return', 'annualized_return', 'sharpe_ratio',
                'sortino_ratio', 'max_drawdown', 'fitness']

    final_df = final_df[cols_order]

    # Display as comparison table
    print("\n=== Genetic Algorithm Chromosome Performance Comparison ===")
    print(final_df.to_string(index=False))

    # Save to CSV for further analysis
    final_df.to_csv('data/GA_chromosome_metrics.csv', index=False)
    print("\nResults saved to GA_chromosome_metrics.csv")

    return None