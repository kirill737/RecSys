import torch
import torch.nn as nn
import torch.sparse as tsparse
from data import df, n_movies, n_users
import numpy as np
import pandas as pd
from logger_settings import setup_logger
import streamlit as st
from data import add_interaction, add_user, movie_index2title, interacted_movies_df

logger = setup_logger("model")
logger.info("Creating model...")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LightGCNModel(nn.Module):
    

    def __init__(self, data, n_users: int, n_movies: int, n_layers: int = 3, latent_dim: int = 64, r: int = 0.5, device=None):
        """
        Args:
            data (pandas.Dataframe): таблица с 'user_id_index' and 'movie_id_index'
            n_users: кол-во пользователей 
            n_movies: кол-во фильмов и сериалов
            n_layers: кол-во слоёв в сети
            latent_dim (int): размерность вектора эмбеддингов
            r: r-AdjNorm параметр (r > 0)
        """
        super(LightGCNModel, self).__init__()   
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.r = r
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        # print(f'Задаём начальные эмбеддинги...')
        self._init_embedding()
        # print(f'Вычисляем матрицу A с крышкой...')
        self.norm_adj_mat_sparse_tensor = self._get_A_tilda_mat(data).to(self.device)
        print(f"Модель создана: r={r}")

    
    def _init_embedding(self):
        """
        Начальная инициализация эмбеддингов
        """
        self.E0 = nn.Embedding(self.n_users + self.n_movies, self.latent_dim).to(self.device)
        nn.init.xavier_uniform_(self.E0.weight)
        self.E0.weight = nn.Parameter(self.E0.weight).to(self.device)


    def update_adj_mat(self, new_data):
        """
        Обновляет матрицу взаимодействий без переобучения эмбеддингов.
        Args:
            new_data (pandas.DataFrame): новые взаимодействия user_index, movie_index
        """
        print("Добавляем новые взаимодейсвия в модель...")
        self.norm_adj_mat_sparse_tensor = self._get_A_tilda_mat(new_data).to(self.device)


    def add_new_users(self, n_new_users: int, interactions_df: pd.DataFrame = None):
        """
        Добавляет новых пользователей в модель без переобучения.
        
        Args:
            n_new_users (int): количество новых пользователей.
            interactions_df (pd.DataFrame): необязательно — взаимодействия новых пользователей (user_index, movie_index).
                                            Здесь индексы должны быть уже в общем пространстве модели (т.е. >= старого n_users).
        """
        print(f"Добавление {n_new_users} новых пользователей...")

        old_n_users = self.n_users
        self.n_users += n_new_users
        new_user_indices = list(range(old_n_users, self.n_users))
        add_user(new_user_indices)
        # Расширяем эмбеддинги
        with torch.no_grad():
            new_weights = torch.empty(n_new_users, self.latent_dim).to(self.device)
            nn.init.xavier_uniform_(new_weights)
            updated_weights = torch.cat([self.E0.weight.data, new_weights], dim=0)
            self.E0 = nn.Embedding(self.n_users + self.n_movies, self.latent_dim).to(self.device)
            self.E0.weight = nn.Parameter(updated_weights)
        
        # Обновляем матрицу A, если есть взаимодействия
        if interactions_df is not None and not interactions_df.empty:
            print("Взаимодействия есть")
            self.update_adj_mat(interactions_df)
            print(interactions_df)
            titles = []
            movie_ids = []
            for _, row in interactions_df.iterrows():
                user_index = row['user_index']
                movie_indices = row['movie_index']  # список индексов фильмов
                print("MOVIES INDEXES:", movie_indices)
                # selected_titles = movie_index2title.get(movie_indices, "Unknown Title") 
                # titles.append(selected_titles)
                movie_ids.append(movie_indices)
            add_interaction(user_index, movie_ids)
        else:
            print("Взаимодействий нет")
            dummy_data = pd.DataFrame({
                "user_index": [],
                "movie_index": []
            })
            self.update_adj_mat(dummy_data)

        print(f"Теперь в модели {self.n_users} пользователей.")
        print(f"Добавлены пользователи с индексами: {new_user_indices}")


    def _get_A_tilda_mat(self, data):
        """
        Получение матрицы A с крышкой
        """
        # Индексы и значения взаимодействий
        indices = torch.LongTensor(np.array([data['user_index'].values, data['movie_index'].values]))
        values = torch.FloatTensor([1.0] * len(data))
    
        R = torch.sparse.FloatTensor(indices, values, torch.Size([self.n_users, self.n_movies]))
        
        l_u_indices = torch.LongTensor(np.array([data['user_index'].values, data['user_index'].values]))
        l_u_values = torch.FloatTensor([0.0] * len(data))
        left_up_zero = torch.sparse.FloatTensor(l_u_indices, l_u_values, torch.Size([self.n_users, self.n_users]))

        r_d_indices = torch.LongTensor(np.array([data['movie_index'].values, data['movie_index'].values]))
        r_d_values = torch.FloatTensor([0.0] * len(data))
        right_down_zero = torch.sparse.FloatTensor(r_d_indices, r_d_values, torch.Size([self.n_movies, self.n_movies]))
        
        # Собираем матрицу A (разреженную)
        upper_part = torch.cat((left_up_zero, R), dim=1)
        lower_part = torch.cat((R.t(), right_down_zero), dim=1)
        A = torch.vstack((upper_part, lower_part))
    
        # Вычисляем степени вершин
        rowsum = A.sum(1)
        
        offsets = torch.zeros((1,), dtype=torch.long)
        # D^(-r)
        D_left = torch.pow(1e-9 + rowsum.to_dense(), -self.r)
        D_left[torch.isinf(D_left)] = 0
        D_left_mat = torch.sparse.spdiags(
                diagonals=D_left,
                offsets=offsets,
                shape=(self.n_users + self.n_movies, self.n_users + self.n_movies)
            )
    
        # D^-(1-r)
        D_right = torch.pow(1e-9 + rowsum.to_dense(), -(1 - self.r))
        D_right[torch.isinf(D_right)] = 0
        D_right_mat = torch.sparse.spdiags(
                diagonals=D_right,
                offsets=offsets,
                shape=(self.n_users + self.n_movies, self.n_users + self.n_movies)
            )
    
        # Собираем нормализованную A: D^(-r) * A * D^-(1-r)
        norm_A = torch.sparse.mm(torch.sparse.mm(D_left_mat, A), D_right_mat)
        return norm_A


    def _propagate_through_layers(self):
        """
        Процес свёртки по L слоям в LightGCN
        """
        E_lyr = self.E0.weight
        device = E_lyr.device
        adj = self.norm_adj_mat_sparse_tensor.to(device)
    
        all_embeddings = [E_lyr]
    
        for _ in range(self.n_layers):
            E_lyr = torch.sparse.mm(adj, E_lyr)
            all_embeddings.append(E_lyr)
    
        # Среднее по всем слоям (включая начальный)
        E_lyr = torch.stack(all_embeddings, dim=0).mean(dim=0)
        final_user_Embed, final_movie_Embed = torch.split(E_lyr, [self.n_users, self.n_movies])
    
        return final_user_Embed, final_movie_Embed


    def forward(self, users, pos_movies, neg_movies):
        """
        Args:
            users (Tensor (batch_size,)):
                tensor of users' indices in batch
            pos_movies (Tensor (batch_size,)):
                tensor of positive movies' indices in batch
            neg_movies (Tensor (batch_size * m,)):
                tensor of negative movies' indices in batch;
                There is m negative movies for every (user - pos_movie) pair

        Returns: tuple of
            usr_embeds (Tensor (batch_size, latent_dim)):
                Updated batch of averaged over layers users' embeddings
            pos_embeds (Tensor (batch_size, latent_dim)):
                Updated batch of averaged over layers positive movies' embeddings 
            neg_embeds (Tensor (batch_size * m, latent_dim)):
                Updated batch of averaged over layers negative movies' embeddings
            initial_usr_embeds (Tensor (batch_size, latent_dim)):
                Batch of users' embeddings on the first layer
            initial_pos_embeds (Tensor (batch_size, latent_dim)):
                Batch of positive movies' embeddings on the first layer
            initial_neg_embeds (Tensor (batch_size * m, latent_dim)):
                Batch of negative movies' embeddings on the first layer
        """
        user_embeds, movie_embeds = self._propagate_through_layers()
        user_embeds, pos_embeds = user_embeds[users], movie_embeds[pos_movies]
        neg_embeds = movie_embeds[neg_movies]

        return user_embeds, pos_embeds, neg_embeds
    

hyperparams = {
    # model params
    'latent_dim': 64,
    'n_layers': 3,
    'r': 0.75,

    # optimizer params
    'patience': 1,
    'learning_rate': 2e-5,
    'decay': 1e-5,

    # training params
    'batch_size': 4096 * 25,
    'neg_samples_for_user': 10, # Кол-во негативных примеров на положительный
    'temperature': 0.5,  # Температура модели
    'epochs': 10
}


@st.cache_data
def create_model():
    logger.info("Downloading fine tunned model...")
    state_dict = torch.load('./models/upd_model_SSM_th_5_r_0.75_t_0.5_weights.pth', map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v

    logger.info("Createing LightGCN model...")
    model = LightGCNModel(
        data=df,
        n_users=n_users,
        n_movies=n_movies,
        n_layers=hyperparams["n_layers"],
        latent_dim=hyperparams["latent_dim"],
        r=hyperparams["r"],
        device=device
    )
    logger.info("Model created!")
    model.load_state_dict(new_state_dict)
    return model


model = create_model()

