import TransGeneralEmbeddingToOPT as ge
import optuna


def findBest(dataset_path):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: ge.find_best_hypers(trial, dataset_path), n_trials=100)

    # Print best hyperparameters
    print(study.best_params)


# # Songs dataset
# findBest("resources/datasets/up_to_ten_tokens_dataset.pt")

# wiki dataset
findBest("resources/datasets/up_to_ten_tokens_dataset_wiki_5.pt")
