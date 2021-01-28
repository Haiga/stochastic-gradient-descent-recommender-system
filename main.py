import sys
import numpy as np
from cf_utils import indexTwoSets
from io_utils import readFile, writePredict
from models import SGD

np.random.seed(2021)

if __name__ == '__main__':
    all_users_predict, all_items_predict, users_items_array_to_predict = readFile(sys.argv[2], type="test",
                                                                                  type_return="array")
    all_users_train, all_items_train, users_items_map_train = readFile(sys.argv[1], type="train")

    """
        Chamadas de métodos de CF - não utilizamos essa abordagem pois tem alta complexidade 
    """
    # indexes_users, indexes_items, ratings, bool_ratings = createMatrixRatings(all_users_predict,
    #                                                                           all_items_predict,
    #                                                                           all_users_train,
    #                                                                           all_items_train,
    #                                                                           users_items_map_train)
    # ratings = computMeanCenteringNormalization(ratings, bool_ratings)

    # similarities = []
    # for user in all_users_test:
    #     similarities.append(computeSimilarity(indexes_users[user], m))
    #     break

    """
        Indexamos os usuários e items para saber suas posições nas matrizes P e Q do SGD
    """
    indexes_users, users_without_rating = indexTwoSets(all_users_train, all_users_predict)
    indexes_items, _ = indexTwoSets(all_items_predict, all_items_train)

    """
        Melhores parâmetros obtidos no processo de tuning
    """

    alpha = 0.0007
    lambda_reg = 0.001
    num_latent_factor = 8
    num_iteracoes = 30

    model = SGD(num_iteracoes=num_iteracoes, num_latent_factor=num_latent_factor, alpha=alpha, lambda_reg=lambda_reg)
    model.fit(users_items_map_train, indexes_users, indexes_items, all_users_predict, all_items_predict)
    predicts = model.predict(users_items_array_to_predict, indexes_users, indexes_items)

    writePredict("i-results.csv", users_items_array_to_predict, predicts, verbose=False)
