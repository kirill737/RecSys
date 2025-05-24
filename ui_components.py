import streamlit as st
import pandas as pd
from state_manager import update_page, get_user_state
from logger_settings import setup_logger
from state_manager import get_selected_user_index, add_rel, add_unrel
from data import user_index2id, movie_id2index, title2movie_index, add_interaction, remove_interaction
from models.LightGCN import model


logger = setup_logger("ui")


def render_sidebar(users_df, interactions_df, movie_index2title):
    logger.info(f"Rendering sidebar...")

    user_name = st.session_state.selected_user_name 
    user_index = st.session_state.selected_user_index

    if user_index is None or user_name is None:
        st.sidebar.header(f"Пользователь не выбран")
        return

    user_state = get_user_state()
    if user_state is None:
        st.sidebar.warning("Нет состояния пользователя")
        return

    st.sidebar.header(f"Информация пользователя {user_name}")
    logger.info(f"Selected user: '{user_name, user_index}'")



    st.sidebar.markdown("---")



    # Меню добавление отслеживаемого пользователя
    st.sidebar.subheader("Добавить отслеживаемого пользователя")
    follow_input = st.sidebar.text_input("Введите индекс пользователя", key="follow_user_input")

    if st.sidebar.button("Добавить в отслеживаемые", key="add_followed_user"):
        if not follow_input.isdigit():
            st.sidebar.warning("Введите корректный числовой индекс.")
        else:
            follow_index = int(follow_input)

            if follow_index not in users_df['user_index'].values:
                st.sidebar.error("Такого пользователя не существует.")
            elif follow_index == user_index:
                st.sidebar.info("Нельзя отслеживать самого себя.")
            else:
                # Обновление users_df
                row_idx_users = users_df.index[users_df['user_index'] == user_index][0]
                user_list_users = users_df.at[row_idx_users, 'user_indices']
                if follow_index not in user_list_users:
                    user_list_users.append(follow_index)
                    users_df.at[row_idx_users, 'user_indices'] = user_list_users
                else:
                    st.sidebar.info("Пользователь уже в списке отслеживаемых.")
                
                if user_index in st.session_state.followed_users_df['user_index'].values:
                    row_idx_followed = st.session_state.followed_users_df.index[st.session_state.followed_users_df['user_index'] == user_index][0]
                    user_list_followed = st.session_state.followed_users_df.at[row_idx_followed, 'user_indices']
                    if follow_index not in user_list_followed:
                        logger.info("followed_users_df update")
                        logger.debug(f"Adding {follow_index}")
                        user_list_followed.append(follow_index)
                        st.session_state.followed_users_df.at[row_idx_followed, 'user_indices'] = user_list_followed
                        logger.debug(st.session_state.followed_users_df.tail(5))
                        st.sidebar.success(f"Пользователь {follow_index} добавлен в отслеживаемые.")
                    else:
                        st.sidebar.info("Пользователь уже в списке отслеживаемых.")
                else:
                    new_row = pd.DataFrame({
                        'user_index': [user_index],
                        'user_indices': [[follow_index]]
                    })
                    st.session_state.followed_users_df = pd.concat([st.session_state.followed_users_df, new_row], ignore_index=True)
                    st.sidebar.success(f"Пользователь {follow_index} добавлен в отслеживаемые (новый).")



    st.sidebar.markdown("---")
    


    # Меню добавления нового понравившегося фильмы 
    st.sidebar.subheader("Добавить понравившийся фильм")
    # new_user_movie_input = st.sidebar.text_input("Введите индекс пользователя")


    def update_new_user_movie():
        new_user_movie = st.session_state.new_user_movie
        selected_user_index = st.session_state.selected_user_index
        logger.info(f"User '{selected_user_index}' adding movie : '{new_user_movie}'")
    

    movie_titles_and_index = [(title, index) for index, title in movie_index2title.items()]
    

    st.sidebar.selectbox(
        "Добавить понравившийся фильм",
        movie_titles_and_index,
        format_func=lambda x: x[0],
        key="new_user_movie",
        on_change=update_new_user_movie
    )


    with st.sidebar:
        col_add, col_rem = st.columns([1, 1])

        with col_add:
            # st.markdown(f"**Add**")
            if st.button("Добавить"):
                logger.info(f"Adding movie '{1}' to user '{1}'")
                new_data = pd.DataFrame({
                    "user_index": [st.session_state.selected_user_index],
                    "movie_index": [st.session_state.new_user_movie[1]]
                })
                add_interaction(user_index, [st.session_state.new_user_movie[1]])
                model.update_adj_mat(new_data)
                # st.rerun()

        with col_rem:
            # st.markdown(f"**Del**")
            if st.button("Удалить"):
                logger.info(f"Removing movie '{1}' from user '{1}'")
                # new_data = pd.DataFrame({
                #     "user_index": [st.session_state.selected_user_index],
                #     "movie_index": [st.session_state.new_user_movie[1]]
                # })
                remove_interaction(user_index, [st.session_state.new_user_movie[1]])
                # st.rerun()



    st.sidebar.markdown("---")



    try:
        seen = interactions_df.loc[user_index, 'movie_index']
        seen_titles = [movie_index2title.get(mid, str(mid)) for mid in seen]
        st.sidebar.write("Просмотренные фильмы:", seen_titles)

        followed = users_df[users_df['user_index'] == user_index].values[0][1]
        st.sidebar.write("Отслеживаемые пользователи:", followed)

        avoid_titles = [movie_index2title.get(mid, str(mid)) for mid in user_state['avoid']]
        st.sidebar.write("Фильмы которые не нужно рекомендовать:", avoid_titles)

        follow_titles = [movie_index2title.get(mid, str(mid)) for mid in user_state['follow_recs']]
        st.sidebar.write("Фильмы от отслеживаемых пользователей:", follow_titles)

        liked_titles = [movie_index2title.get(mid, str(mid)) for mid in user_state['rel']]
        st.sidebar.write("Likes:", liked_titles)

        disliked_titles = [movie_index2title.get(mid, str(mid)) for mid in user_state['unrel']]
        st.sidebar.write("Dislikes:", disliked_titles)

    except Exception as e:
        st.sidebar.write("Нет данных")
        logger.warning(f"Sidebar data fetch error: {e}")

    logger.info(f"Sidebar rendered!")


def render_recommendations(movie_index2title):
    logger.info(f"Rendering recs...")

    user_state = get_user_state()

    if user_state is None:
        st.warning("Нет состояния пользователя")
        return

    recs = user_state['recommendations']
    page = user_state['page']

    if not recs:
        st.info("Нет рекомендаций. Нажмите кнопку.")
        logger.info("No recs")
        return

    st.subheader("Рекомендации:")
    for i, rec in enumerate(recs[page*10:(page+1)*10]):
        col_title, col_buttons, col_is_follow = st.columns([6, 2, 1])

        with col_title:
            st.markdown(f"**{movie_index2title.get(rec, f'Movie {rec}')}**")

        with col_buttons:
            rel_col, unrel_col = st.columns([1, 1])

            with rel_col:
                if st.button("👍", key=f"rel_{rec}_{page}_{i}"):
                    logger.info(f"'like' film {rec}")
                    user_state['rel'].add(rec)
                    st.rerun()

            with unrel_col:
                if st.button("👎", key=f"unrel_{rec}_{page}_{i}"):
                    logger.info(f"'dislike' film {rec}")
                    user_state['unrel'].add(rec)
                    st.rerun()

        with col_is_follow:
            if rec in user_state['follow_recs']:
                st.markdown("+")
            else:
                st.markdown("")

    col1, page_col, col2 = st.columns(3)
    with col1:
        if st.button("← Назад", key="prev_page") and page > 0:
            update_page(-1)
            st.rerun()

    with page_col:
        st.markdown(f"Страница {page + 1}")

    with col2:
        if st.button("Далее →", key="next_page") and (page + 1) * 10 < len(recs):
            update_page(1)
            st.rerun()
