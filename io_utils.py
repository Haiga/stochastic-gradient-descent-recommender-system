import re

"""
Lemos os arquivos de treino e teste de uma só vez para garantir que todos os usuários e itens da coleção estarão
na matriz de ratings
"""

def readFile(name_file, type="train", ignore_first_line=True, users_to_add=None, items_to_add=None,
             intersect_or_union='intersect', type_return='dict'):
    """
    Essa função lê um arquivo de usuários itens, com ou sem ratings e timestamp
    Para arquivos do tipo train é retornado um set com os usuários, um set com os itens, e um terceiro retorno
    Se type_return for 'dict' o terceiro retorno é um dicionário com chaves para usuários, e cada usuário contém
    um dicionário para seus itens e cada item indexa o rating do usuário chave para tal item
    Se type_return for 'array' o terceiro retorno é um array com pares de user,item (type == test) ou trios de
    user,item,rating (type == train)

    Parameters
    ----------
    name_file : str
        O nome do arquivo a ser lido.

    type : {'train', 'test'}, optional
        O tipo do arquivo a ser lido.
            Se 'train' cada linha do arquivo deve ter o formato u{user_id}:i{item_id},{rating},{timestamp}
            Se 'test' cada linha do arquivo deve ter o formato u{user_id}:i{item_id}

    ignore_first_line : boolean, optional
        Se True ignora a primeira linha do arquivo a ser lido

    users_to_add: set or dict, optional
        Se esse parâmetro for utilizado observe também os parâmetros items_to_add e intersect_or_union
        A saída irá conter apenas usuários em users_to_add
            Pode conter usuários que não estão em users_to_add se (items_to_add != None & intersect_or_union == 'union')

    items_to_add: set or dict, optional
        Se esse parâmetro for utilizado observe também os parâmetros items_to_add e intersect_or_union
        A saída irá conter apenas itens em items_to_add
            Pode conter itens que não estão em items_to_add se (users_to_add != None & intersect_or_union == 'union')

    intersect_or_union : {'intersect', 'union'}, optional
        Esse parâmetro só é utilizado quando users_to_add e items_to_add são ambos não nulos
        Se 'intersect' a saída ignora as linhas com usuários que não estão em users_to_add & ignora as linhas com itens que
            não estão em items_to_add.
        Se 'union' a saída ignora as linhas com usuários que não estão em users_to_add a menos que o item da linha
            contenha algum item em items_to_add. Ignora também as linhas com items que não estão em items_to_add a menos
            que o usuário da linha contenha algum usuário em users_to_add.

    type_return : {'array', 'dict'}, optional
        Esse parâmetro indica se o terceiro item do retorno é um array ou um dicionário
        Se type_return for 'dict' o terceiro retorno é um dicionário com chaves para usuários, e cada usuário contém
        um dicionário para seus itens e cada item indexa o rating do usuário chave para tal item
        Se type_return for 'array' o terceiro retorno é um array com pares de user,item (type == test) ou trios de
        user,item,rating (type == train)

    """

    with open(name_file, 'r') as file:
        lines = file.readlines()

    if ignore_first_line:
        lines = lines[1:]

    if type == "test":
        p = re.compile('u([0-9]*):i([0-9]*)')
    elif type == "train":
        p = re.compile('u([0-9]*):i([0-9]*),([0-9]*),([0-9]*)')
    else:
        raise Exception("Invalid type of file")

    all_users = set()
    all_items = set()

    users_items_map = {}

    cont_removed_ratings = 0

    add_all = users_to_add is None and items_to_add is None
    add_by_items = users_to_add is None and not items_to_add is None
    add_by_users = not users_to_add is None and items_to_add is None

    array_user_item_pairs = []

    for line in lines:
        m = p.match(line)

        u = int(m.groups()[0])
        i = int(m.groups()[1])

        if type == "test":
            r = 0
        else:
            r = int(m.groups()[2])

        all_users.add(u)
        all_items.add(i)

        should_be_added = False
        if add_all:
            should_be_added = True
        elif add_by_items and add_by_users:
            if intersect_or_union == 'intersect':
                if u in users_to_add and i in items_to_add:
                    should_be_added = True
            if intersect_or_union == 'union':
                if u in users_to_add or i in items_to_add:
                    should_be_added = True
        elif add_by_users:
            if u in users_to_add:
                should_be_added = True
        elif add_by_items:
            if i in items_to_add:
                should_be_added = True

        if should_be_added:
            if type_return == 'array' and type == 'train':
                array_user_item_pairs.append((u, i, r))
            elif type_return == 'array' and type == 'test':
                array_user_item_pairs.append((u, i))
            else:
                users_items_map.setdefault(u, {})
                users_items_map[u].setdefault(i, r)
        else:
            cont_removed_ratings += 1

    # print(f"{cont_removed_ratings} removed ratings")
    if type_return == 'array':
        return all_users, all_items, array_user_item_pairs
    return all_users, all_items, users_items_map


def writePredict(name_file_output, users_itens, predicts, header="UserId:ItemId,Prediction", verbose=False, max_punctuation=10, min_punctuation=0):
    """

    Parameters
    ----------
    name_file_output: str Nome do arquivo de saída
    users_itens: array de pares (u, i)
    predicts: array de ratings preditos para todos os pares (u, i)
    header: str Cabeçalho do arquivo de saída
    verbose: bool Escreve as predições na saída padrão
    max_punctuation: int, optional Se alguma predição fica acima de max_punctuation, a predição é convertida para esse máximo
    min_punctuation: int, optional Se alguma predição fica abaixo de min_punctuation, a predição é convertida para esse mínimo

    Returns
    -------

    """
    if verbose:
        for u_i, p in zip(users_itens, predicts):
            u = u_i[0]
            i = u_i[1]
            if p > max_punctuation:
                print("u" + str(u).zfill(7) + ":" + "i" + str(i).zfill(7) + "," + str(10))
            elif p < min_punctuation:
                print("u" + str(u).zfill(7) + ":" + "i" + str(i).zfill(7) + "," + str(0))
            else:
                print("u" + str(u).zfill(7) + ":" + "i" + str(i).zfill(7) + "," + str(p))
    else:
        with open(name_file_output, 'w') as file:
            file.write(header + "\n")
            for u_i, p in zip(users_itens, predicts):
                u = u_i[0]
                i = u_i[1]
                if p > max_punctuation:
                    file.write("u" + str(u).zfill(7) + ":" + "i" + str(i).zfill(7) + "," + str(10) + "\n")
                elif p < min_punctuation:
                    file.write("u" + str(u).zfill(7) + ":" + "i" + str(i).zfill(7) + "," + str(0) + "\n")
                else:
                    file.write("u" + str(u).zfill(7) + ":" + "i" + str(i).zfill(7) + "," + str(p) + "\n")