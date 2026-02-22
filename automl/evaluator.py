def select_best_model(results):
    best_model = max(results, key=lambda k: results[k]["score"])
    return best_model, results[best_model]