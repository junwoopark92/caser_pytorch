import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import activation_getter


class Caser(nn.Module):
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

    def __init__(self, num_users, num_items, model_args):
        super(Caser, self).__init__()
        self.args = model_args

        # init args
        L = self.args.L
        dims = self.args.d
        self.n_h = self.args.nh
        self.n_v = self.args.nv
        self.drop_ratio = self.args.drop
        self.ac_conv = activation_getter[self.args.ac_conv]
        self.ac_fc = activation_getter[self.args.ac_fc]

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, dims)
        self.item_embeddings = nn.Embedding(num_items, dims)

        # vertical conv layer
        self.conv_v = nn.Conv2d(1, self.n_v, (L, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(L)]
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, dims)) for i in lengths])

        # fully-connected layer
        self.fc1_dim_v = self.n_v * dims
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        # W1, b1 can be encoded with nn.Linear
        self.fc1 = nn.Linear(fc1_dim_in, dims)
        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(num_items, dims+dims)
        self.b2 = nn.Embedding(num_items, 1)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

        self.cache_x = None

    def forward(self, seq_var, user_var, item_var, use_cache=False, for_pred=False):
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
            item_embs = self.item_embeddings(seq_var).unsqueeze(1)  # use unsqueeze() to get 4-D
            user_emb = self.user_embeddings(user_var).squeeze(1)

            # Convolutional Layers
            out, out_h, out_v = None, None, None
            # vertical conv layer
            if self.n_v:
                out_v = self.conv_v(item_embs).squeeze(2)
                out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

            # horizontal conv layer
            out_hs = list()
            if self.n_h:
                for conv in self.conv_h:
                    conv_out = self.ac_conv(conv(item_embs).squeeze(3))
                    pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                    out_hs.append(pool_out)
                out_h = torch.cat(out_hs, 1)  # prepare for fully connect

            # Fully-connected Layers
            out = torch.cat([out_v, out_h], 1)
            # apply dropout
            out = self.dropout(out)

            # fully-connected layer
            z = self.ac_fc(self.fc1(out))
            x = torch.cat([z, user_emb], 1)

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

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        if mask:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


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
        self.user_embeddings = nn.Embedding(num_users, dims)
        self.item_embeddings = nn.Embedding(num_items, dims)
        self.category_embeddings = nn.Embedding(topic_num, dims)

        self.multihead_attn = MultiHeadAttention(5, 50, 50, 50, dropout=0.5)

        # fully-connected layer
        # W1, b1 can be encoded with nn.Linear
        self.fc1 = nn.Linear(dims + dims, dims)
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
            get_sinusoid_encoding_table(n_position, dims, None),
            freeze=True)

    def attn_layer(self, q):
        q_T = q.transpose(1, 2)
        qq_T = torch.bmm(q, q_T)
        attn_map = torch.sum(qq_T, dim=1).unsqueeze(2)
        return F.softmax(attn_map, dim=1)

    def scaledot_attn_layer(self, q):
        q_T = q.transpose(1, 2)
        qq_T = torch.bmm(q, q_T)
        attn_map = torch.sum(qq_T, dim=1).unsqueeze(2)
        attn_map = attn_map / self.temperature
        attn_map = F.softmax(attn_map, dim=1)
        attn_map = self.dropout(attn_map)
        return attn_map

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
            user_emb = self.user_embeddings(user_var).squeeze(1)

            q = item_embs + self.position_enc.weight

            attn_repeat = 2
            for i in range(attn_repeat):
                # attn_map = self.attn_layer(q)
                # attn_map = self.scaledot_attn_layer(q)
                q, attn_map = self.multihead_attn(q, q, q)

                # q = item_embs * attn_map

            # print(q.shape)
            item_seq_vec = q.sum(dim=1)
            # print(item_seq_vec.shape)
            # categorical vector
            category_embeddings = self.category_embeddings.weight
            categorical_vector = probs @ category_embeddings
            if categorical_vector.dim() == 1:
                categorical_vector = categorical_vector.reshape(1, self.dims)

            out = torch.cat([item_seq_vec, categorical_vector], dim=1)

            # apply dropout
            out = self.dropout(out)

            # fully-connected layer
            z = self.ac_fc(self.fc1(out))
            x = torch.cat([z, user_emb], 1)

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
