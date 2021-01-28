import numpy as np
np.random.seed(2021)

class SGD:
    """
    SGD - Stochastic gradient descent
    Pode ser aplicada como uma técnica de fatorização de matrizes de ratings em sistemas de recomendação
    """

    def __init__(self, alpha=0.001, num_iteracoes=50, num_latent_factor=5, lambda_reg=0):
        """

        Parameters
        ----------
        alpha: float Taxa de aprendizado do SGD - regula o tamanho do passo do gradiente
        num_iteracoes: int Quantidade de iterações performadas
            (cada iteração ajusta cada um dos ratings - pointwise - em users_items_map_train
        num_latent_factor: int número de fatores latentes que serão estimados
        lambda_reg: float regularaização do erro - para evitar overfitting uma vez que é aprendido uma representação
            a partir de dados muito esparsos
        """
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.num_iteracoes = num_iteracoes
        self.num_latent_factor = num_latent_factor
        self.P = []
        self.Q = []

    def fit(self, users_items_map_train, indexes_users, indexes_items, users_keys_test=None, items_keys_test=None):
        """

        Parameters
        ----------
        users_items_map_train: dict of dicts dicionário de usuários, onde cada usuário tem um dicionário de itens mapeando
            o rating desse usuário para esse item
        indexes_users: dict Dicionário com todos os usuários que serão utilizados para construir as matrizes de fatores
            latentes (pode conter users not in users_items_map_train)
        indexes_items: dict Dicionário com todos os itens que serão utilizados para construir as matrizes de fatores
            latentes (pode conter items not in users_items_map_train)
        users_keys_test: dict Dicionário com usuários que serão focados no processo de treino, se None todos os usuários
            em users_items_map_train serão igualmente avaliados
        items_keys_test: dict Dicionário com items que serão focados no processo de treino, se None todos os items
            em users_items_map_train serão igualmente avaliados
        Returns
        -------
        Retorna as matrizes P e Q que são representações aproximadas da matriz de ratings

        """
        num_users = len(indexes_users)
        num_items = len(indexes_items)

        if users_keys_test is None:
            users_keys_test = indexes_users
        if items_keys_test is None:
            items_keys_test = indexes_items

        self.P = np.ones((num_users, self.num_latent_factor), dtype=np.float)
        self.Q = np.ones((self.num_latent_factor, num_items), dtype=np.float)

        user_keys = np.array(list(users_items_map_train.keys()))

        for iter in range(self.num_iteracoes):
            # Adicionando uma aleatorização na forma com que os usuários são percorridos para evitar ficar preso em minimos locais
            # np.random.shuffle(user_keys)
            for u in user_keys:
                for i in users_items_map_train[u]:
                    if u in users_keys_test or i in items_keys_test:
                        eui = users_items_map_train[u][i] - np.matmul(
                            (self.P[indexes_users[u]]), (self.Q[:, indexes_items[i]]))

                        rP = self.P[indexes_users[u]] + 2 * self.alpha * eui * self.Q[:, indexes_items[
                                                                                             i]] - 2 * self.alpha * self.lambda_reg * \
                             self.P[indexes_users[u]]

                        stop = False
                        if np.isnan(rP).any():
                            stop = True

                        rQ = self.Q[:,
                             indexes_items[i]] + 2 * self.alpha * eui * rP - 2 * self.alpha * self.lambda_reg * \
                             - 2 * self.alpha * self.lambda_reg * self.Q[:, indexes_items[i]]

                        if np.isnan(rQ).any():
                            stop = True

                        # O método pode divergir para valores muito grandes de alpha
                        if not stop:
                            self.P[indexes_users[u]] = rP
                            self.Q[:, indexes_items[i]] = rQ
                        else:
                            return self.P, self.Q
        return self.P, self.Q

    def predict(self, users_itens, indexes_users, indexes_items, users_without_rating=None,
                type_of_predict_for_users_without_rating='default'):
        """

        Parameters
        ----------
        users_itens: array of (u, i) para ser computado o rating do user u para o item i
        indexes_users: dict Dicionário com todos os usuários que serão utilizados para construir as matrizes de fatores
            latentes (pode conter users not in users_items_map_train) - contém os índices dos usuários na matriz de fator latente
        indexes_items: dict Dicionário com todos os itens que serão utilizados para construir as matrizes de fatores
            latentes (pode conter items not in users_items_map_train) - contém os índices dos itens na matriz de fator latente
        users_without_rating: dict Dicionário com todos os usuários que não possuem ratings no conjunto de treino (fit function)
            Para esses usuários o rating para um item i pode ser a média de ratings desse item i, a média global de ratings,
            ou o valor da decomposição SGD - que é especificado no parâmetro type_of_predict_for_users_without_rating
        type_of_predict_for_users_without_rating: {'mean_item', 'global_mean', 'default'}, optional
            mean_item - Para usuários do conjunto users_without_rating o rating computado para um item i é a média dos ratings do item i
            global_mean - Para usuários do conjunto users_without_rating o rating computado para um item i é a média global dos ratings
            default - Utiliza a decomposição do SGD para gerar todos os ratings
        Returns
        -------
             Retorna um array de ratings para cada par (u, i) em users_itens
        """

        if not users_without_rating is None:
            print("inside")
            result = np.matmul(self.P, self.Q)
            mean_item = np.mean(result, axis=0)
            global_mean = np.mean(mean_item)
        else:
            users_without_rating = {}
            result = None

        predicts = np.zeros(len(users_itens), dtype=np.float)
        cont = 0

        if users_without_rating is None:
            users_without_rating = {}

        for (u, i) in users_itens:
            # for (u, i, r) in users_itens:
            if u in users_without_rating:
                if type_of_predict_for_users_without_rating == 'mean_item':
                    predicts[cont] = mean_item[indexes_items[i]]
                elif type_of_predict_for_users_without_rating == 'global_mean':
                    predicts[cont] = global_mean
                else:
                    predicts[cont] = result[indexes_users[u]][indexes_items[i]]
                #
            else:
                if result is None:
                    predicts[cont] = np.matmul(self.P[indexes_users[u]], self.Q[:, indexes_items[i]])
                else:
                    predicts[cont] = result[indexes_users[u]][indexes_items[i]]
            cont += 1
        return predicts
