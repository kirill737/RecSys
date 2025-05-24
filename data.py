import pandas as pd
from logger_settings import setup_logger
import streamlit as st
logger = setup_logger("data")

logger.info("Preparing data...")

@st.cache_data
def load_data():
    movies_dataset = pd.read_csv('../../dataset/ml-32m/movies.csv')
    movies_dataset['genres_list'] = movies_dataset['genres'].apply(lambda s: s.split('|'))
    all_genres = set(movies_dataset['genres_list'].apply(lambda s: s[0]).unique())
    movies_dataset['some_genre'] = movies_dataset['genres_list'].apply(lambda x: x[0])
    genre_name2index = {name: index for index, name in enumerate(all_genres)}
    movie_id2genre_index = {
        movie_id: genre_name2index[name] for (movie_id, name) in
        zip(movies_dataset['movieId'].values, movies_dataset['some_genre'].values)
    }
    movie_id2title = dict(zip(movies_dataset['movieId'], movies_dataset['title']))
    title2movie_id = dict(zip(movies_dataset['title'], movies_dataset['movieId']))
    
    df = pd.read_pickle("../dfs/last_train_th_5_df.pkl")
    interacted_movies_df = df.groupby("user_index")["movie_index"].apply(list).reset_index()
    interacted_movies_df = interacted_movies_df.set_index('user_index')

    
    followed_users_df = pd.read_json("./prepared_data/followed_users.json", orient="records", lines=True)
    
    followed_users_df = followed_users_df.rename(columns={"users_index": "user_index"})
    followed_users_df = followed_users_df.rename(columns={"followed_users_indicies": "user_indices"})
    if 'followed_users_df' not in st.session_state:
        st.session_state.followed_users_df = followed_users_df
    return interacted_movies_df, df, movie_id2genre_index, movie_id2title, title2movie_id

interacted_movies_df, df, movie_id2genre_index, movie_id2title, title2movie_id = load_data()


def add_follows():
    pass

user_ids = df['user_id'].unique() # Множество U
user_id2index = {id: index for index, id in enumerate(user_ids)}
user_index2id = {index: id for id, index in user_id2index.items()}

movie_ids = df['movie_id'].unique() # Множество V
movie_id2index = {id: index for index, id in enumerate(movie_ids)}
movie_index2id = {index: id for id, index in movie_id2index.items()}

movie_index2title = {index: movie_id2title.get(movie_id, "Unknown Title") 
                     for movie_id, index in movie_id2index.items()}

title2movie_index = {movie_id2title.get(movie_id, "Unknown Title"): index 
                     for movie_id, index in movie_id2index.items()}


n_users = len(user_ids) # M - кол-во пользователей
n_movies = len(movie_ids) # N - кол-во фильмов

logger.info("Data prepared!")


def load_users():
    return st.session_state.followed_users_df


def add_user(new_users: list[str]):
    """
    Добавляет новых пользователей в followed_users_df.

    Args:
        new_users (list[str]): список новых имён (user_id).
    """
    print("Добавляем нового пользователя...")
    global df, user_id2index, user_index2id, user_ids, n_users
    # print("Step 0", new_users)
    
    for new_user in new_users:
        # print(f"Пользователя с id: '{new_user}'")
        # if user_index2id[new_user] in user_id2index:
        #     logger.warning(f"Пользователь '{new_user}' уже существует.")
        #     continue
        # print("Step 1")
        # Назначаем новый индекс
        new_index = n_users
        user_id2index[new_user] = new_index
        user_index2id[new_index] = new_user
        user_ids = list(user_id2index.keys())  # обновляем множество
        n_users += 1
        # print("Step 2: old len = ", len(followed_users_df))
        # Добавляем пустой список подписок (user_indices)
        new_row = {"user_index": new_index, "user_indices": []}
        st.session_state.followed_users_df = pd.concat([st.session_state.followed_users_df, pd.DataFrame([new_row])], ignore_index=True)
        # print("Step 3: new len = ", len(followed_users_df)

        logger.info(f"Добавлен пользователь '{new_user}' с индексом {new_index}")


def load_interactions():
    return interacted_movies_df


def add_interaction(user_index: int, selected_title_indices: list[str]):
    """
    Добавляет взаимодействия (просмотры фильмов) для нового пользователя.

    Args:
        user_index (int): индекс нового пользователя.
        selected_title_indices (list[str]): список названий фильмов.
    """
    logger.debug(f"Adding inteaction for user_index: {user_index}, selected_title_indices:{selected_title_indices}")
    global interacted_movies_df, df, movie_id2index

    new_movie_indices = []
    for movie_index in selected_title_indices:
        # movie_index = title2movie_index.get(title)
        if movie_index is not None:
            new_movie_indices.append(movie_index)
        else:
            logger.warning(f"Movie '{movie_index2title[movie_index]}' not found in dict.")

    if not new_movie_indices:
        logger.warning(f"No valid movie for user_index{user_index}")
        return

    # Добавляем взаимодействия в основной датафрейм (df)
    new_rows = pd.DataFrame({
        "user_index": [user_index] * len(new_movie_indices),
        "movie_index": new_movie_indices
    })
    df = pd.concat([df, new_rows], ignore_index=True)

    # Обновляем inter
    # Добавить, а не заменить
    if user_index in interacted_movies_df.index:
        old_movies = interacted_movies_df.loc[user_index][0]  # или .iloc[0] если это Series
    else:
        old_movies = []

    logger.debug("Old movies")
    logger.debug(old_movies)

    logger.debug("Interactions")
    logger.debug(interacted_movies_df)
    updated_movies = list(set(old_movies + new_movie_indices))  # можно без set, если повторы допустимы
    interacted_movies_df.loc[user_index] = [updated_movies]


    logger.info(f"Add {len(new_movie_indices)} new interactions for {user_index}")

def remove_interaction(user_index: int, selected_title_indices: list[str]):
    """
    Удаляет взаимодействия (просмотры фильмов) для пользователя.

    Args:
        user_index (int): индекс пользователя.
        selected_title_indices (list[str]): список индексов фильмов для удаления.
    """
    logger.debug(f"Removing interaction for user_index: {user_index}, selected_title_indices:{selected_title_indices}")
    global interacted_movies_df, df, movie_id2index

    remove_movie_indices = []
    for movie_index in selected_title_indices:
        if movie_index is not None:
            remove_movie_indices.append(movie_index)
        else:
            logger.warning(f"Movie '{movie_index2title[movie_index]}' not found in dict.")

    if not remove_movie_indices:
        logger.warning(f"No valid movies to remove for user_index {user_index}")
        return

    # Удаляем строки из df, соответствующие user_index и movie_index
    initial_len = len(df)
    df = df[~((df["user_index"] == user_index) & (df["movie_index"].isin(remove_movie_indices)))]
    logger.info(f"Removed {initial_len - len(df)} interactions from df for user_index {user_index}")

    # Обновляем interacted_movies_df
    if user_index in interacted_movies_df.index:
        current_movies = interacted_movies_df.loc[user_index][0]
        updated_movies = [idx for idx in current_movies if idx not in remove_movie_indices]

        if updated_movies:
            interacted_movies_df.loc[user_index] = [updated_movies]
        else:
            # Удаляем строку полностью, если не осталось фильмов
            interacted_movies_df = interacted_movies_df.drop(index=user_index)

        logger.info(f"Updated interactions for user_index {user_index}")
    else:
        logger.warning(f"No existing interactions for user_index {user_index}")



user_index2name = {}
user_name2index = {}


def add_user_name(user_index, user_name):
    user_index2name[user_index] = user_name
    user_name2index[user_name] = user_index


def get_user_name(user_index):
    if user_index in user_index2name:
        return user_index2name[user_index]
    return user_index


def get_user_index(user_name):
    if user_name in user_name2index:
        return user_name2index[user_name]
    return user_name

