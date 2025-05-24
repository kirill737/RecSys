import numpy as np
import pandas as pd
from state_manager import get_user_state, set_recommendations, get_selected_user_index, set_avoid, set_norm_recs, set_follow_recs, get_avoid
from models.LightGCN import model, hyperparams, LightGCNModel
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm.auto import tqdm
import random
import streamlit as st
tqdm_notebook.pandas()
import torch
import gc


from collections import OrderedDict

from data import user_id2index, movie_id2index, interacted_movies_df

from logger_settings import setup_logger

logger = setup_logger("recommender")


def generate_new_recs(
    model, selected_user_indices,
    temp: float, k: int = 10,
    batch_size: int = 1000,
    user_index2id: dict = user_id2index,
    movie_index2id: dict = movie_id2index,
    interacted_movies_df=None, avoid_df=None
):
    logger.info("Start generating recs...")
    """
    Генерирует top-k рекомендации для пользователей (в виде DataFrame),
    исключая уже просмотренные и ранее рекомендованные фильмы.
    """
    logger.debug("Calc interactions...")
    # Словари взаимодействий и игнор-листа
    interactions_dict = {}
    if interacted_movies_df is not None and not interacted_movies_df.empty:
        interactions_dict = {
            user_index: set(row['movie_index'])
            for user_index, row in interacted_movies_df.iterrows()
        }

    avoid_dict = {} if avoid_df is None or avoid_df.empty else {
        row["user_index"]: set(row['avoid_movie_indices']) for _, row in avoid_df.iterrows()
    }

    # Эмбеддинги
    model.eval()
    with torch.no_grad():
        user_embeds, movie_embeds = model._propagate_through_layers()
        selected_user_embeds = user_embeds[selected_user_indices].detach()

    del user_embeds
    gc.collect()
    torch.cuda.empty_cache()

    n_users = selected_user_embeds.shape[0]
    n_batch = n_users // batch_size
    residue = n_users % batch_size

    selected_user_embeds /= selected_user_embeds.norm(dim=-1, keepdim=True)
    movie_embeds /= movie_embeds.norm(dim=-1, keepdim=True)

    top_k_indecies_list = []
    max_score_global = -float("inf")
    min_score_global = float("inf")
    logger.debug("Gen cycle start...")
    for batch_index in range(n_batch + (1 if residue > 0 else 0)):
        start_index = batch_index * batch_size
        finish_index = start_index + (residue if batch_index == n_batch else batch_size)
        user_embeds_batch = selected_user_embeds[start_index:finish_index]

        relevance_score = torch.matmul(user_embeds_batch, torch.transpose(movie_embeds, 0, 1))
        relevance_score /= temp

        max_score_global = max(max_score_global, relevance_score.max().item())
        min_score_global = min(min_score_global, relevance_score.min().item())

         # Обнуляем баллы просмотренных и ранее рекомендованных фильмов
        for i, user_index in enumerate(selected_user_indices[start_index:finish_index]):
            watched = interactions_dict.get(user_index, set())
            avoided = avoid_dict.get(user_index, set())
            to_mask = watched.union(avoided)
            if to_mask:
                relevance_score[i, list(to_mask)] = -1e9

        top_k = torch.topk(relevance_score, k=k, dim=1)[1]
        top_k_indecies_list.append(top_k)

    logger.debug("Recs cycle complited")
    print(f"\nМинимальный полученный Score: {min_score_global:.4f}")
    print(f"Максимальный полученный Score: {max_score_global:.4f}\n")

    top_k_movie_indecies = torch.vstack(top_k_indecies_list)

    recs_df = pd.DataFrame({
        'user_index': selected_user_indices,
        'movie_indices': [
            top_k_movie_indecies[i].cpu().numpy().tolist()
            for i in range(len(selected_user_indices))
        ]
    })
    logger.debug("Updating avoid list...")
    # Обновлённый avoid_list
    updated_avoid = {}
    for i, user_index in enumerate(selected_user_indices):
        old_avoided = avoid_dict.get(user_index, set())
        new_avoided = old_avoided.union(set(top_k_movie_indecies[i].cpu().numpy()))
        updated_avoid[user_index] = [int(x) for x in new_avoided]

    updated_avoid_df = pd.DataFrame([
        {"user_index": user_index, "avoid_movie_indices": movie_indices}
        for user_index, movie_indices in updated_avoid.items()
    ])

    del top_k_movie_indecies, movie_embeds
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Recs generated!")
    return recs_df, updated_avoid_df


def gen_follow_recs(selected_user_indices: list, k: int, followed_users_df: pd.DataFrame, 
                    interacted_movies_df: pd.DataFrame, avoid_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Генерация рекомендаций на основе фильмов, просмотренных отслеживаемыми пользователями,
    исключая уже просмотренные и ранее рекомендованные фильмы.

    Возвращает:
        pd.DataFrame с колонками:
            - 'user_index'
            - 'movies_indices' (список индексов фильмов)
    """
    follow_recs = []

    for user_index in selected_user_indices:
        try:
            print("---------> follow cols", st.session_state.followed_users_df.columns)
            followed_users = st.session_state.followed_users_df.loc[
                st.session_state.followed_users_df['user_index'] == user_index, 'user_indices'
            ].values[0]
        except IndexError:
            followed_users = []

        movie_set = set()
        for f_user in followed_users:
            if f_user in interacted_movies_df.index:
                movie_set.update(interacted_movies_df.loc[f_user, 'movie_index'])

        # Исключаем уже просмотренные фильмы
        watched_by_user = set(interacted_movies_df.loc[user_index, 'movie_index']) if user_index in interacted_movies_df.index else set()
        movie_set -= watched_by_user

        # Исключаем фильмы из avoid_df, если есть
        if avoid_df is not None and not avoid_df.empty and user_index in avoid_df.index:
            avoid_by_user = set(avoid_df.loc[user_index, 'avoid_movie_indices'])
            movie_set -= avoid_by_user

        movie_list = list(movie_set)
        recs = random.sample(movie_list, min(k, len(movie_list))) if movie_list else []

        follow_recs.append({
            "user_index": user_index,
            "movie_indices": recs
        })

    return pd.DataFrame(follow_recs)


def generate_recommendations():
    logger.info("Generating recs...")
    user_index = get_selected_user_index()
    # recs = list(np.random.choice(range(100, 200), 25))
    old_avoid_df = pd.DataFrame([{
        'user_index': st.session_state.selected_user_index,
        'avoid_movie_indices': get_user_state()["avoid"] or []
    }])
    if user_index in interacted_movies_df.index:
        interacted_movies_df.at[user_index, 'movie_index'] += get_user_state()['rel'] or []
    
    logger.info("Generating norm recs...")
    norm_k_recs, new_avoid_df = generate_new_recs(
        model=model,
        selected_user_indices=[user_index],
        temp=hyperparams['temperature'],
        k=90,
        batch_size=100,
        interacted_movies_df=interacted_movies_df,
        avoid_df=old_avoid_df
        )
    # logger.debug("norm_k_recs cols", norm_k_recs.columns)
    logger.debug(f"norm_k_recs")
    logger.debug(norm_k_recs)
    print("\t\t---> Generating follow recs")
    logger.info("Generating follow recs...")
    logger.info("Follow df")
    logger.debug(st.session_state.followed_users_df.tail(5))
    follow_k_recs = gen_follow_recs(
        selected_user_indices=[user_index],
        k=10,
        followed_users_df=st.session_state.followed_users_df,
        interacted_movies_df=interacted_movies_df,
        avoid_df=new_avoid_df
    )
    logger.debug("follow_k_recs")
    logger.debug(follow_k_recs)
    follow_dict = follow_k_recs.set_index("user_index")["movie_indices"].to_dict()
    new_avoid_df["avoid_movie_indices"] = new_avoid_df.apply(
        lambda row: list(set(row["avoid_movie_indices"]) | set(follow_dict.get(row["user_index"], []))),
        axis=1
    )


    norm_df = norm_k_recs.rename(columns={"movie_indices": "norm_recs"})
    follow_df = follow_k_recs.rename(columns={"movie_indices": "follow_recs"})
    logger.info("Merging recs...")
    # Объединяем по user_index
    merged = pd.merge(norm_df, follow_df, on="user_index", how="outer")
    logger.debug("merged", merged)
    print("---------> COLUMNS: ", merged.columns)
    # Заполняем пустые значения пустыми списками
    merged["norm_recs"] = merged["norm_recs"].apply(lambda x: x if isinstance(x, list) else [])
    merged["follow_recs"] = merged["follow_recs"].apply(lambda x: x if isinstance(x, list) else [])

    # Объединяем списки
    merged["merged_recs"] = merged.apply(lambda row: row["norm_recs"] + row["follow_recs"], axis=1)
    merged['merged_recs'] = merged['merged_recs'].apply(lambda x: random.sample(x, len(x)))

    print("recs", merged[["user_index", "merged_recs"]].values[0][1])
    print("recs cols", merged.columns)
    print("len final", len(merged))
    set_avoid(new_avoid_df.values[0][1])
    set_norm_recs(norm_k_recs.values[0][1])
    set_follow_recs(follow_k_recs.values[0][1])

    set_recommendations(merged[["user_index", "merged_recs"]].values[0][1])


def add_new_user_with_interactions(
    new_user_id: int,
    interacted_movie_ids: list,
    user_id2index: dict,
    movie_id2index: dict,
    model: LightGCNModel
):
    global interacted_movies_df
    """
    Добавляет нового пользователя, его взаимодействия, и обновляет модель.

    Аргументы:
    - new_user_id: ID нового пользователя (уникальный)
    - interacted_movie_ids: список ID фильмов, с которыми пользователь взаимодействовал
    - user_id2index: словарь ID → index
    - movie_id2index: словарь ID фильма → index
    - interacted_movies_df: текущий DataFrame взаимодействий
    - model: обученная модель LightGCN
    """
    if new_user_id in user_id2index:
        raise ValueError(f"Пользователь {new_user_id} уже существует")

    new_user_index = max(user_id2index.values()) + 1
    user_id2index[new_user_id] = new_user_index

    # Добавление взаимодействий
    valid_movie_indices = [
        movie_id2index[movie_id] for movie_id in interacted_movie_ids if movie_id in movie_id2index
    ]

    # Обновление interacted_movies_df
    interacted_movies_df.loc[new_user_index] = pd.Series({
        "user_index": new_user_index,
        "movie_index": valid_movie_indices
    })

    

    # Обновление числа пользователей в модели
    model.num_users += 1

    # При необходимости можно обновить граф взаимодействий
    model.rebuild_adj_mat(interacted_movies_df)  # если у вас реализован метод `rebuild_adj_mat`

    return new_user_index



