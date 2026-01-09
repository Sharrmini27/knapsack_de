import numpy as np

def run_de(data, pop_size, generations, F, CR):
    n_items = 500
    # Initialize Population (values between 0 and 1)
    pop = np.random.rand(pop_size, n_items)
    history = []

    for gen in range(generations):
        new_pop = []
        for i in range(pop_size):
            # Mutation: V = X_r1 + F * (X_r2 - X_r3)
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), 0, 1)
            
            # Crossover
            trial = np.where(np.random.rand(n_items) < CR, mutant, pop[i])
            
            # Binary Mapping & Repair (Constraint w1)
            binary_sol = (trial > 0.5).astype(int)
            total_w1 = np.sum(binary_sol * data['w1'])
            
            if total_w1 > data['capacity']:
                # Repair: Remove items until under capacity
                on_idxs = np.where(binary_sol == 1)[0]
                for idx in on_idxs:
                    binary_sol[idx] = 0
                    total_w1 -= data['w1'][idx]
                    if total_w1 <= data['capacity']: break
            
            new_pop.append(trial)
            
        pop = np.array(new_pop)
        
        # Calculate best profit in this generation for the convergence curve
        current_best_profit = max([np.sum((p > 0.5).astype(int) * data['values']) for p in pop])
        history.append(current_best_profit)

    return history, pop
