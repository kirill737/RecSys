import streamlit as st
from logger_settings import setup_logger

logger = setup_logger("state_manager")


def get_user_state():
    user_index = st.session_state.get("selected_user_index")
    if user_index is None:
        return None

    key = f"user_state_{user_index}"
    if key not in st.session_state:
        st.session_state[key] = {
            'recommendations': [],
            'page': 0,
            'avoid': set(),
            'follow_recs': [],
            'norm_recs': [],
            'rel': set(),
            'unrel': set()
        }
    return st.session_state[key]


def set_selected_movie(movie_index):
    user_state = get_user_state()
    user_state['selected_movie'] = movie_index


def get_selected_movie():
    user_state = get_user_state()
    return user_state['selected_movie']


def update_page(offset: int):
    user_state = get_user_state()
    if user_state:
        user_state['page'] += offset


def set_recommendations(recs):
    user_state = get_user_state()
    if user_state is not None:
        user_state['recommendations'] = recs
        user_state['page'] = 0


def set_avoid(recs):
    user_state = get_user_state()
    if user_state is not None:
        logger.info("Setting avoid")
        user_state['avoid'] = recs


def get_avoid():
    user_state = get_user_state()
    return user_state['avoid'] if user_state else None


def set_norm_recs(recs):
    user_state = get_user_state()
    if user_state is not None:
        user_state['norm_recs'] = recs


def set_follow_recs(recs):
    logger.debug(f"Set follow recs to:")
    logger.debug(recs)
    user_state = get_user_state()
    if user_state is not None:
        user_state['follow_recs'] = recs


def add_rel(movie_index):
    user_state = get_user_state()
    if user_state is not None:
        user_state['rel'].add(movie_index)
        user_state['unrel'].discard(movie_index)


def add_unrel(movie_index):
    user_state = get_user_state()
    if user_state is not None:
        user_state['unrel'].add(movie_index)
        user_state['rel'].discard(movie_index)


def get_selected_user_index():
    return st.session_state.get("selected_user_index")
