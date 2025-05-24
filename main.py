import streamlit as st
import pandas as pd
from data import (
    load_users, load_interactions, movie_index2title, title2movie_index,
    user_name2index, user_index2name, add_user_name, get_user_name,
    interacted_movies_df, get_user_index
)
from recommender import generate_recommendations
from state_manager import (
    get_user_state, set_recommendations, set_avoid,
    set_follow_recs, set_norm_recs
)
from ui_components import render_sidebar, render_recommendations
from logger_settings import setup_logger
from models.LightGCN import model

logger = setup_logger("main")

# ====== Загрузка данных ======
users_df = load_users()
interactions_df = load_interactions()

# ====== Интерфейс ======
st.header("ВКР | Рекомендательная система")

# ====== Управление состоянием ======
if "show_add_user_form" not in st.session_state:
    st.session_state.show_add_user_form = False
if "selected_movies" not in st.session_state:
    st.session_state.selected_movies = set()
if "show_movie_selector" not in st.session_state:
    st.session_state.show_movie_selector = False
if "selected_user_index" not in st.session_state:
    st.session_state.selected_user_index = None
if "new_user_movie" not in st.session_state:
    st.session_state.new_user_movie = None
if "followed_users_df" not in st.session_state:
    st.session_state.followed_users_df = None


# ====== Кнопка для отображения формы добавления пользователя ======
if st.button("Добавить пользователя"):
    st.session_state.show_add_user_form = True
    st.session_state.selected_movies = set()  # сброс выбора


# ====== Меню добавления пользователя ======
if st.session_state.show_add_user_form:
    st.subheader("Добавление нового пользователя")
    new_user_name = st.text_input("Имя пользователя")

    # Кнопка для открытия выбора фильмов
    if st.button("Добавить просмотренный фильм"):
        st.session_state.show_movie_selector = not st.session_state.show_movie_selector

    # Меню выбора фильмов
    if st.session_state.show_movie_selector:
        st.markdown("**Выберите понравившиеся фильмы:**")
        movie_titles = list(movie_index2title.values())
        search_query = st.text_input("Поиск по названию")

        # Фильтрация по поисковому запросу
        filtered_movies = [title for title in movie_titles if search_query.lower() in title.lower()]

        # Выбор фильмов с чекбоксами
        for title in filtered_movies:
            if st.checkbox(title, key=title):
                st.session_state.selected_movies.add(title)
            else:
                st.session_state.selected_movies.discard(title)

    # Отображение текущего выбора
    st.markdown(f"Вы выбрали {len(st.session_state.selected_movies)} фильмов:")
    if st.session_state.selected_movies:
        st.write(list(st.session_state.selected_movies))

    # Кнопка сохранения нового пользователя
    if st.button("Сохранить пользователя"):
        if not new_user_name:
            st.warning("Пожалуйста, введите имя пользователя.")
        elif len(st.session_state.selected_movies) < 5:
            st.warning("Пожалуйста, выберите минимум 5 фильмов.")
        else:
            new_user_index = model.n_users
            add_user_name(new_user_index, new_user_name)

            selected_movie_indices = [
                title2movie_index[title] for title in st.session_state.selected_movies
            ]

            new_interactions = pd.DataFrame({
                "user_index": [new_user_index] * len(selected_movie_indices),
                "movie_index": selected_movie_indices
            })

            model.add_new_users(n_new_users=1, interactions_df=new_interactions)
            if new_user_index not in interacted_movies_df.index:
                interacted_movies_df.loc[new_user_index] = [selected_movie_indices]
            else:
                st.warning("Пользователь уже есть в таблице взаимодействий.")

            st.success(f"Пользователь '{new_user_name}' добавлен с {len(selected_movie_indices)} фильмами.")

            # Сброс состояний
            st.session_state.show_add_user_form = False
            st.session_state.show_movie_selector = False
            st.session_state.selected_movies = set()
            st.rerun()

# ====== Выбор пользователя ======
def update_selected_user_index():
    selected_user_name = st.session_state.selected_user_name
    selected_user_index = get_user_index(selected_user_name)
    st.session_state.selected_user_index = selected_user_index
    logger.info(f"Select user index: {selected_user_index}, name: {selected_user_name}")


users_list = [get_user_name(index) for index in users_df['user_index'][190000:]]
st.selectbox(
    "Выберите пользователя",
    users_list,
    key="selected_user_name",
    on_change=update_selected_user_index
)

# ====== Боковая панель ======
render_sidebar(users_df, interactions_df, movie_index2title)

# ====== Генерация рекомендаций ======
if st.session_state.get("selected_user_index") is not None:
    user_state = get_user_state()
    st.button("Сгенерировать рекомендации", on_click=generate_recommendations)
    render_recommendations(movie_index2title)
