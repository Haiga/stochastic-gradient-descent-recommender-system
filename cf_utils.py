import numpy as np
np.random.seed(2021)

def indexTwoSets(first_set, second_set):
    """
    Retorna a indexação dos dados nos conjuntos first_set e second_set (essa indexação pode ser utilizada para acessar a
    posição de um elemento do conjunto em um vetor indexado a partir de zero

    Parameters
    ----------
    first_set: set or dict
        É um conjunto de dados únicos
    second_set: set or dict
        É um conjunto de dados únicos

    Returns
    -------
        indexes, not_in_first_set
        indexes:
            Junta os dados do first_set com o second_set, e para cada item única desse conjunto é assinalado um índice
        not_in_first_set: dict
            Contém as chaves que estão no second_set mas não no first_set

    """

    indexes = {}
    count = 0
    for u in first_set:
        if u not in indexes:
            indexes.setdefault(u, count)
            count += 1

    not_in_first_set = {}
    for u in second_set:
        if u not in indexes:
            indexes.setdefault(u, count)
            count += 1
            not_in_first_set.setdefault(u, 0)

    return indexes, not_in_first_set


"""
Nos comentários utilizaremos o id do usuário para definir o valor que aparece no arquivo de ratings
O índice do usuário é a linha na matriz (de ratings) que o usuário está
Raciocínio análogo para os itens

Operações vetorizadas são muito rápidas, 
porém como os dados são extremamente esparsos a computação vetorizada não foi ""eficiente""

"""

def createMatrixRatings(all_users_test, all_items_test, all_users_train, all_items_train,
                        users_itens_map_train, default_value=0):
    """
    Retorna a indexação de todos os usuários em all_users_test e all_users_train, assim como indexa todos os itens
    em all_items_test e all_items_train na matriz de ratings (np.array bidimensional) que também é retornada

    Parameters
    ----------
    all_users_test: set or dict
        É um conjunto de usuários, é necessário pois não seria possível fazer uma predição para esse usuário se ele não
        estiver na matriz de ratings (mesmo que ele não tenha feito nenhum rating no traino)
    all_items_test: set or dict
        É um conjunto de itens, é necessário pois não seria possível fazer uma predição desse item para um usuário se
        ele não estiver na matriz de ratings (mesmo que ele não tenha recebido nenhum rating no traino)
    all_users_train: set or dict
        É o conjunto de usuários em users_itens_map_train
    all_items_train: set or dict
        É o conjunto de items em users_itens_map_train
    users_itens_map_train: dict
        Contém ratings de usuários para itens: As chaves iniciais são usuários, e cada usuário contém um dicionário
        para seus itens, e cada item indexa o rating do usuário chave para tal item
    default_value: float
        É o valor padrão dos ratings na matriz retornada para os pares de (usuário, item) que não tem rating

    Returns
    -------
        indexes_users, indexes_items, ratings, bool_ratings
        indexes_users:
            É um dicionário com a união das chaves em all_users_train e all_users_test, e mapeia o índice desse usuário
            na matriz de ratings (é um índice de uma linha da matriz)
        indexes_items:
            É um dicionário com a união das chaves em all_items_train e all_items_test, e mapeia o índice desse item
            na matriz de ratings (é um índice de uma coluna da matriz)
        ratings:
            É um np.array bidimensinal onde cada linha é um usuário (de all_users_train e all_users_test)
            e cada coluna um item de (de all_items_train e all_items_test), com valor padrão definido em default_value,
            e com os valores dos ratings de users_itens_map_train
        bool_ratings:
            É um np.array bidimensinal onde cada linha é um usuário (de all_users_train e all_users_test)
            e cada coluna um item de (de all_items_train e all_items_test), com valor padrão 0,
            e onde há ratings em users_itens_map_train o valor é 1 (apenas para diferenciar um rating 0 do valor default 0)

    """

    indexes_users = indexTwoSets(all_users_train, all_users_test)
    user_count = len(indexes_users)

    indexes_items = indexTwoSets(all_items_test, all_items_train)
    item_count = len(indexes_items)

    ratings = np.zeros((user_count, item_count), dtype=np.float) + default_value
    bool_ratings = np.zeros((user_count, item_count), dtype=np.float)
    # i_bool_ratings = np.ones((user_count, item_count), dtype=np.float)

    for u in users_itens_map_train:
        for i in users_itens_map_train[u]:
            ratings[indexes_users[u]][indexes_items[i]] = users_itens_map_train[u][i]
            bool_ratings[indexes_users[u]][indexes_items[i]] = 1
            # i_bool_ratings[indexes_users[u]][indexes_items[i]] = 0

    return indexes_users, indexes_items, ratings, bool_ratings


def computMeanCenteringNormalization(ratings, bool_ratings):
    """

    Parameters
    ----------
    ratings: np.array
        Matriz de ratings (cada linha é um usuário e cada coluna um item)
    bool_ratings: np.array
        Matriz boolean de ratings (cada linha é um usuário e cada coluna um item),
        É utilizada para diferenciar um rating 0 de um valor 0 na matriz que é a ausência de rating
    Returns
    -------
        Retorna uma matriz do mesmo tamanho das matriz de entrada, após fazer o cálculo da normalização central de rating
    """
    users_mean_rating = np.sum(ratings, axis=1) / np.sum(bool_ratings, axis=1)
    np.nan_to_num(users_mean_rating, copy=False)
    num_items = ratings.shape[1]
    return ratings - np.multiply(bool_ratings, np.tile(users_mean_rating, (num_items, 1)).T)


def computeCosineSimilarity(user_index, ratings):
    """

    Parameters
    ----------
    user_index: int
        Índice do usuário na matriz de ratings para o qual queremos calcular a similaridade com os demais usuários da matriz
    ratings: np.array
        Matriz de ratings (cada linha é um usuário e cada coluna um item)

    Returns
    -------
        Retorna um np.array com tamanho igual ao número de linhas da matriz de ratings, onde cada valor desse vetor é
        a similaridade do cosseno usuário dessa linha na matriz com o usuário do índice passado em user_index
    """
    all_similarities = np.sum(ratings[user_index] * ratings, axis=1) / (np.sqrt(np.sum(
        ratings[user_index] * ratings[user_index])) * np.sqrt(np.sum(np.multiply(ratings, ratings), axis=1)))
    return all_similarities
