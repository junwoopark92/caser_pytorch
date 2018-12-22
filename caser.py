import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import activation_getter


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv
        if mask:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


class SelfAttnCaser(nn.Module):
    """
    Convolutional Sequence Embedding Recommendation Model (Caser)[1].
    [1] Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, Jiaxi Tang and Ke Wang , WSDM '18
    Parameters
    ----------
    num_users: int,
        Number of users.
    num_items: int,
        Number of items.
    model_args: args,
        Model-related arguments, like latent dimensions.
    """

    def __init__(self, num_users, num_items, model_args, topic_num):
        super(SelfAttnCaser, self).__init__()
        self.args = model_args

        # init args
        L = self.args.L
        dims = self.args.d
        self.dims = dims
        self.L = L
        self.drop_ratio = self.args.drop
        self.ac_fc = activation_getter[self.args.ac_fc]

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, dims, max_norm=1.0)
        self.item_embeddings = nn.Embedding(num_items, dims, max_norm=1.0)
        self.category_embeddings = nn.Embedding(topic_num, dims, max_norm=1.0)
        self.embedding_dropout = nn.Dropout(0.1)

        n_layers = 3
        self.layer_stack = nn.ModuleList([
            EncoderLayer(50, 50, 8, 50, 50, dropout=0.1) for _ in range(n_layers)])

        # fully-connected layer
        # W1, b1 can be encoded with nn.Linear
        # self.fc1 = nn.Linear(dims + dims, dims)
        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(num_items, dims + dims)
        self.b2 = nn.Embedding(num_items, 1)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.category_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

        self.cache_x = None

        n_position = L

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, dims, None), freeze=True)

    def forward(self, seq_var, user_var, item_var, probs, use_cache=False, for_pred=False):
        """
        The forward propagation used to get recommendation scores, given
        triplet (user, sequence, targets). Note that we can cache 'x' to
        save computation for negative predictions. Because when computing
        negatives, the (user, sequence) are the same, thus 'x' will be the
        same as well.
        Parameters
        ----------
        seq_var: torch.autograd.Variable
            a batch of sequence
        user_var: torch.autograd.Variable
            a batch of user
        item_var: torch.autograd.Variable
            a batch of items
        use_cache: boolean, optional
            Use cache of x. Set to True when computing negatives.
        for_pred: boolean, optional
            Train or Prediction. Set to True when evaluation.
        """

        if not use_cache:
            # Embedding Look-up
            item_embs = self.item_embeddings(seq_var)
            item_embs = self.embedding_dropout(item_embs)
            user_emb = self.user_embeddings(user_var).squeeze(1)
            user_emb = self.embedding_dropout(user_emb)

            q = item_embs + self.position_enc.weight
            q_ = q

            for enc_layer in self.layer_stack:
                q, attn_map = enc_layer(q)
                q = q + q_
                q_ = q

            item_seq_vec = q.sum(dim=1)

            # categorical vector
            category_embeddings = self.category_embeddings.weight
            category_embeddings = self.embedding_dropout(category_embeddings)
            categorical_vector = probs @ category_embeddings
            if categorical_vector.dim() == 1:
                categorical_vector = categorical_vector.reshape(1, self.dims)

            """
            out = torch.cat([item_seq_vec, categorical_vector], dim=1)
            out = self.dropout(out)

            # fully-connected layer
            x = self.ac_fc(self.fc1(out))
            """

            gate = torch.sigmoid((categorical_vector * user_emb).sum(dim=1)).unsqueeze(1)
            x = gate * categorical_vector + (1 - gate) * item_seq_vec
            x = torch.cat([x, user_emb], dim=1)

            self.cache_x = x

        else:
            x = self.cache_x

        w2 = self.W2(item_var)
        b2 = self.b2(item_var)
        if not for_pred:
            results = []
            for i in range(item_var.size(1)):
                w2i = w2[:, i, :]
                b2i = b2[:, i, 0]
                result = (x * w2i).sum(1) + b2i
                results.append(result)
            res = torch.stack(results, 1)
        else:
            w2 = w2.squeeze()
            b2 = b2.squeeze()
            res = (x * w2).sum(1) + b2

        return res