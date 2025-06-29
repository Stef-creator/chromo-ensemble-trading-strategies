import numpy as np
import pandas as pd
import random
from copy import deepcopy


def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        chromosome = [
            random.uniform(5, 40),   # Downtrend Buy value
            random.randint(5, 20),   # Downtrend Buy interval
            random.uniform(60, 95),  # Downtrend Sell value
            random.randint(5, 20),   # Downtrend Sell interval
            random.uniform(5, 40),   # Uptrend Buy value
            random.randint(5, 20),   # Uptrend Buy interval
            random.uniform(60, 95),  # Uptrend Sell value
            random.randint(5, 20)    # Uptrend Sell interval
        ]
        population.append(chromosome)
    return population

def evaluate_fitness(chromosome, df):
    capital = 10000
    position = 0
    capital_series = []

    for _, row in df.iterrows():
        trend = row['Trend']
        close_price = row['Close']

        if trend == 0:
            buy_val, buy_int, sell_val, sell_int = chromosome[:4]
        else:
            buy_val, buy_int, sell_val, sell_int = chromosome[4:]

        rsi_col = f'RSI_{int(buy_int)}'
        if rsi_col in row:
            rsi_value = row[rsi_col]
            if position == 0 and rsi_value < buy_val:
                position = capital / close_price
                capital = 0
            elif position > 0 and rsi_value > sell_val:
                capital = position * close_price
                position = 0

        total_value = capital if position == 0 else position * close_price
        capital_series.append(total_value)

    portfolio = pd.Series(capital_series)
    daily_returns = portfolio.pct_change().dropna()

    total_return = (portfolio.iloc[-1] / portfolio.iloc[0]) - 1
    ann_return = (1 + total_return) ** (252 / len(portfolio)) - 1
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else 0
    downside_std = daily_returns[daily_returns < 0].std()
    sortino = daily_returns.mean() / downside_std * np.sqrt(252) if downside_std != 0 else 0
    cumulative = portfolio.cummax()
    drawdown = (portfolio - cumulative) / cumulative
    max_drawdown = drawdown.min()
    fitness = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        'chromosome': chromosome,
        'total_return': total_return,
        'annualized_return': ann_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'fitness': fitness
    }

def tournament_selection(population, fitnesses, TOURNAMENT_SIZE):
    """
    Tournament selection of a parent.
    """
    participants = random.sample(list(zip(population, fitnesses)), TOURNAMENT_SIZE)
    participants.sort(key=lambda x: x[1]['fitness'], reverse=True)
    return participants[0][0]

def crossover(parent1, parent2, CROSSOVER_PROB):
    """
    Single-point crossover.
    """
    if random.random() < CROSSOVER_PROB:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else:
        return deepcopy(parent1), deepcopy(parent2)

def mutate(chromosome, MUTATION_PROB):
    """
    Random mutation on genes.
    """
    for i in range(len(chromosome)):
        if random.random() < MUTATION_PROB:
            if i % 2 == 0:  # RSI value gene
                chromosome[i] = random.uniform(5, 95) if i % 4 < 2 else random.uniform(60, 95)
            else:  # interval gene
                chromosome[i] = random.randint(5, 20)
    return chromosome

def genetic_algorithm(ticker, POP_SIZE=50, MAX_GENERATIONS=50, ELITISM_COUNT=2, TOURNAMENT_SIZE=3, CROSSOVER_PROB=0.7, MUTATION_PROB=0.001, STAGNATION_LIMIT=10, override_selection_metric=None):
    """
    Full GA loop with early stopping on stagnation.
    """
    df = pd.read_csv(f'data/{ticker}_technical_indicators.csv')
    population = initialize_population(POP_SIZE)
    best_fitness_history = []
    stagnation_counter = 0
    last_best = None

    for generation in range(MAX_GENERATIONS):
        fitnesses = [evaluate_fitness(chromo, df) for chromo in population]
        pop_fitness = list(zip(population, fitnesses))
        pop_fitness.sort(key=lambda x: x[1]['fitness'], reverse=True)
        best = pop_fitness[0][1]['fitness']
        best_fitness_history.append(best)
        print(f"Generation {generation+1}: Best Fitness = {best:.4f}")

        # Check stagnation
        if last_best is not None and abs(best - last_best) < 1e-8:
            stagnation_counter += 1
            print(f"  No improvement. Stagnation count: {stagnation_counter}/{STAGNATION_LIMIT}")
        else:
            stagnation_counter = 0
        last_best = best

        if stagnation_counter >= STAGNATION_LIMIT:
            print("Convergence reached. Early stopping.")
            break

        # Elitism
        new_population = [deepcopy(chromo) for chromo, _ in pop_fitness[:ELITISM_COUNT]]

        while len(new_population) < POP_SIZE:
            parent1 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            child1, child2 = crossover(parent1, parent2, CROSSOVER_PROB)
            new_population.append(mutate(child1, MUTATION_PROB))
            if len(new_population) < POP_SIZE:
                new_population.append(mutate(child2, MUTATION_PROB))

        population = new_population

    # Final evaluation
    final_fitnesses = [evaluate_fitness(chromo, df) for chromo in population]
    results_df = pd.DataFrame([{
        'chromosome': f['chromosome'],
        'total_return': f['total_return'],
        'annualized_return': f['annualized_return'],
        'sharpe_ratio': f['sharpe_ratio'],
        'sortino_ratio': f['sortino_ratio'],
        'max_drawdown': f['max_drawdown'],
        'fitness': f['fitness']
    } for f in final_fitnesses])

    # Expand chromosome genes
    chromo_df = pd.DataFrame(results_df['chromosome'].tolist(),
                            columns=['down_buy_val', 'down_buy_int', 'down_sell_val', 'down_sell_int',
                                    'up_buy_val', 'up_buy_int', 'up_sell_val', 'up_sell_int'])
    final_df = pd.concat([chromo_df, results_df.drop(columns=['chromosome'])], axis=1)

    # If override_selection_metric is provided, use it for selection
    if override_selection_metric and override_selection_metric in final_df.columns:
        final_df.sort_values(override_selection_metric, ascending=False, inplace=True)
        print(f"\nOverride selection applied. Sorted by {override_selection_metric} descending.")
    else:
        # Default sorting by fitness
        final_df.sort_values('fitness', ascending=False, inplace=True)

    # Save results
    final_df.to_csv('data/GA_final_results.csv', index=False)
    print("\nFinal results saved to GA_final_results.csv")