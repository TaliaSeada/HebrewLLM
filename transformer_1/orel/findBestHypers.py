import TransGeneralEmbeddingToOPT as ge
import optuna


study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: ge.find_best_hypers(trial, "resources/up_to_ten_tokens_dataset.pt"), n_trials=100)

# Print best hyperparameters
print(study.best_params)