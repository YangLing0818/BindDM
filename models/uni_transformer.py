import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, knn_graph, GATConv, GCNConv, TopKPooling
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch_scatter import scatter_softmax, scatter_sum
from models.common import GaussianSmearing, MLP, batch_hybrid_edge_connection, outer_product

class BASE(torch.nn.Module):
    def __init__(self,in_channels, sg_channels, ratio=0.5, Conv=GCNConv, non_linearity=torch.tanh, with_sg=False):
        super(BASE,self).__init__()
        self.in_channels = in_channels
        self.sg_channels = sg_channels
        self.with_sg = with_sg
        self.ratio = ratio
        if with_sg:
            self.score_layer = Conv(in_channels + sg_channels, 1)
        else:
            self.score_layer = Conv(in_channels, 1)
        self.non_linearity = non_linearity
    def forward(self, x, edge_index, edge_attr=None, batch=None, subcomplex_pad=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        if self.with_sg:
            assert subcomplex_pad is not None
            score = self.score_layer(torch.cat([x, subcomplex_pad], dim=-1), edge_index).squeeze()
        else:
            score = self.score_layer(x, edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm

class BaseX2HAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r', out_fc=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.ew_net_type = ew_net_type
        self.out_fc = out_fc

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == 'm':
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.hv_func(kv_input)

        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = self.ew_net(v[..., :self.hidden_dim])
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0,
                                dim_size=N)  # [num_edges, n_heads]

        m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
        output = output.view(-1, self.output_dim)
        if self.out_fc:
            output = self.node_output(torch.cat([output, h], -1))

        output = output + h
        return output

class BaseH2XAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.ew_net_type = ew_net_type

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.xv_func(kv_input)
        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = 1.
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)  # (xi - xj) [n_edges, n_heads, 3]
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)  # (E, heads)

        m = alpha.unsqueeze(-1) * v  # (E, heads, 3)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, 3)
        return output.mean(1)  # [num_nodes, 3]

class DualAttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 ew_net_type='r', x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.r_min, self.r_max = r_min, r_max
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)

        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc)
            )

        self.x2h_layers_sub = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers_sub.append(
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc)
            )

        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type)
            )

        self.graph_fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, h, h_sub, x, x_sub, edge_attr, edge_attr_sub, edge_index, edge_index_sub, mask_ligand, perm_sub, e_w=None, e_w_sub=None, fix_x=False):
        src, dst = edge_index
        src_sub, dst_sub = edge_index_sub
        if self.edge_feat_dim > 0:
            edge_feat = edge_attr
            edge_feat_sub = edge_attr_sub
        else:
            edge_feat = None
            edge_feat_sub = None


        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        rel_x_sub = x_sub[dst_sub] - x_sub[src_sub]
        dist_sub = torch.norm(rel_x_sub, p=2, dim=-1, keepdim=True)

        h_in = h
        h_in_sub = h_sub
        for i in range(self.num_x2h):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            h_out = self.x2h_layers[i](h_in, dist_feat, edge_feat, edge_index, e_w=e_w)
            h_in = h_out

        for i in range(self.num_x2h):
            dist_feat_sub = self.distance_expansion(dist_sub)
            dist_feat_sub = outer_product(edge_attr_sub, dist_feat_sub)
            h_out_sub = self.x2h_layers_sub[i](h_in_sub, dist_feat_sub, edge_feat_sub, edge_index_sub, e_w=e_w_sub)
            h_in_sub = h_out_sub

        h_in_sub_pad = torch.zeros_like(h_in).to(h_in.device)
        h_in_sub_pad[perm_sub] = h_in_sub
        h_in = self.graph_fusion(torch.cat([h_in, h_in_sub_pad], dim=-1))

        x2h_out = h_in

        new_h = h if self.sync_twoup else x2h_out
        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            delta_x = self.h2x_layers[i](new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w)
            if not fix_x:
                x = x + delta_x * mask_ligand[:, None]  # only ligand positions will be updated
            rel_x = x[dst] - x[src]
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        return x2h_out, x

class DualAttentionLayerO2TwoUpdateNodeGeneral_interactionnode(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 ew_net_type='r', x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.r_min, self.r_max = r_min, r_max
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)

        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc)
            )

        self.x2h_layers_sub = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers_sub.append(
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc)
            )

        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type)
            )

        self.graph_fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim))
        self.sub_int_fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, h, h_sub, x, x_sub, edge_attr, edge_attr_sub, edge_index, edge_index_sub, mask_ligand, perm_sub, intnode, batch_sub, e_w=None, e_w_sub=None, fix_x=False):
        src, dst = edge_index
        src_sub, dst_sub = edge_index_sub
        if self.edge_feat_dim > 0:
            edge_feat = edge_attr
            edge_feat_sub = edge_attr_sub
        else:
            edge_feat = None
            edge_feat_sub = None

        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        rel_x_sub = x_sub[dst_sub] - x_sub[src_sub]
        dist_sub = torch.norm(rel_x_sub, p=2, dim=-1, keepdim=True)

        h_in = h
        h_in_sub = h_sub

        intnode_pad = intnode.index_select(0, batch_sub)

        h_in_sub = self.sub_int_fusion(torch.cat([h_in_sub, intnode_pad], dim=1)) + h_in_sub

        for i in range(self.num_x2h):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            h_out = self.x2h_layers[i](h_in, dist_feat, edge_feat, edge_index, e_w=e_w)
            h_in = h_out

        for i in range(self.num_x2h):
            dist_feat_sub = self.distance_expansion(dist_sub)
            dist_feat_sub = outer_product(edge_attr_sub, dist_feat_sub)
            h_out_sub = self.x2h_layers_sub[i](h_in_sub, dist_feat_sub, edge_feat_sub, edge_index_sub, e_w=e_w_sub)
            h_in_sub = h_out_sub

        h_in_sub_pad = torch.zeros_like(h_in).to(h_in.device)
        h_in_sub_pad[perm_sub] = h_in_sub
        h_in = self.graph_fusion(torch.cat([h_in, h_in_sub_pad], dim=-1))

        x2h_out = h_in

        new_h = h if self.sync_twoup else x2h_out
        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            delta_x = self.h2x_layers[i](new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w)
            if not fix_x:
                x = x + delta_x * mask_ligand[:, None]  # only ligand positions will be updated
            rel_x = x[dst] - x[src]
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        return x2h_out, x


class UniTransformerO2TwoUpdateGeneral(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, k=32,
                 num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                 cutoff_mode='radius', ew_net_type='r',
                 num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_node_types = num_node_types
        self.cutoff_mode = cutoff_mode  # [radius, none]
        self.k = k
        self.ew_net_type = ew_net_type  # [r, m, none]

        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.num_init_x2h = num_init_x2h
        self.num_init_h2x = num_init_h2x
        self.r_max = r_max
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)
        if self.ew_net_type == 'global':
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)

        self.init_h_emb_layer = self._build_init_h_layer()
        self.base_block = self._build_share_blocks()
        self.bases = self._build_bases(r=0.5)
        self.scmlp = self._build_scmlp()

        # gated transmission module
        self.W_a_main = self._build_W_a_main()
        self.W_a_int = self._build_W_a_int()

        self.W_main = self._build_W_main()
        self.W_bmm = self._build_W_bmm()

        self.W_int = self._build_W_int()
        self.W_main_to_int = self._build_W_main_to_int()
        self.W_int_to_main = self._build_W_int_to_main()

        self.W_zi1 = self._build_W_zi1()
        self.W_zi2 = self._build_W_zi2()

        self.GRU_int = nn.GRUCell(self.hidden_dim, self.hidden_dim)

    def __repr__(self):
        return f'UniTransformerO2(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, ew_net_type={self.ew_net_type}, ' \
               f'init h emb: {self.init_h_emb_layer.__repr__()} \n' \
               f'base block: {self.base_block.__repr__()} \n' \
               f'edge pred layer: {self.edge_pred_layer.__repr__() if hasattr(self, "edge_pred_layer") else "None"}) '

    def _build_init_h_layer(self):
        layer = DualAttentionLayerO2TwoUpdateNodeGeneral(
            self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn, norm=self.norm,
            num_x2h=self.num_init_x2h, num_h2x=self.num_init_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
            ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
        )
        return layer

    def _build_share_blocks(self):
        base_block = []

        base_block.append(DualAttentionLayerO2TwoUpdateNodeGeneral(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn,
                norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
            ))

        for l_idx in range(self.num_layers - 1):
            layer = DualAttentionLayerO2TwoUpdateNodeGeneral_interactionnode(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn,
                norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)

    def _build_bases(self,r=0.5):
        bases = []
        bases.append(BASE(in_channels=self.hidden_dim, sg_channels=self.hidden_dim, ratio=r, with_sg=False))
        for l_idx in range(self.num_layers-1):
            b_ = BASE(in_channels=self.hidden_dim, sg_channels=self.hidden_dim, ratio=r, with_sg=True)
            bases.append(b_)
        return nn.ModuleList(bases)

    def _build_scmlp(self):
        scmlp = []
        for l_idx in range(self.num_layers):
            mlp = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim))
            scmlp.append(mlp)
        return nn.ModuleList(scmlp)

    def _build_W_a_main(self):
        mod = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh()) for i in range(self.num_layers - 1)])
        return mod

    def _build_W_a_int(self):
        mod = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh()) for i in range(self.num_layers - 1)])
        return mod

    def _build_W_main(self):
        mod = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for i in range(self.num_layers - 1)])
        return mod

    def _build_W_bmm(self):
        mod = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for i in range(self.num_layers - 1)])
        return mod

    def _build_W_int(self):
        mod = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh()) for i in range(self.num_layers - 1)])
        return mod

    def _build_W_main_to_int(self):
        mod = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh()) for i in range(self.num_layers - 1)])
        return mod

    def _build_W_int_to_main(self):
        mod = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh()) for i in range(self.num_layers - 1)])
        return mod

    def _build_W_zi1(self):
        mod = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for i in range(self.num_layers - 1)])
        return mod

    def _build_W_zi2(self):
        mod = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for i in range(self.num_layers - 1)])
        return mod

    def _connect_edge(self, x, mask_ligand, batch):
        if self.cutoff_mode == 'radius':
            edge_index = radius_graph(x, r=self.r, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'knn':
            edge_index = knn_graph(x, k=self.k, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'hybrid':
            edge_index = batch_hybrid_edge_connection(
                x, k=self.k, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index

    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1
        n_dst = mask_ligand[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        edge_type = F.one_hot(edge_type, num_classes=4)
        return edge_type

    def forward(self, h, x, mask_ligand, batch, return_all=False, fix_x=False):

        all_x = [x]
        all_h = [h]

        int_node = None
        subcomplex_pad = None

        for b_idx in range(self.num_blocks):
            edge_index = self._connect_edge(x, mask_ligand, batch)
            src, dst = edge_index

            edge_type = self._build_edge_type(edge_index, mask_ligand)
            if self.ew_net_type == 'global':
                dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
                dist_feat = self.distance_expansion(dist)
                logits = self.edge_pred_layer(dist_feat)
                e_w = torch.sigmoid(logits)
            else:
                e_w = None

            for l_idx, layer in enumerate(self.base_block):

                h_sub, edge_index_sub, _, batch_sub, perm_sub = self.bases[l_idx](x=h, edge_index=edge_index, batch=batch, subcomplex_pad=subcomplex_pad)

                mask_ligand_sub = mask_ligand[perm_sub]
                x_sub = x[perm_sub]
                edge_type_sub = self._build_edge_type(edge_index_sub, mask_ligand_sub)

                if l_idx == 0:
                    int_node = scatter_sum(h_sub, batch_sub, dim=0)
                    h, x = layer(h, h_sub, x, x_sub, edge_type, edge_type_sub, edge_index, edge_index_sub, mask_ligand, perm_sub, e_w=e_w, e_w_sub=None, fix_x=fix_x)

                else:
                    assert int_node is not None

                    a_sub = self.W_a_main[l_idx - 1](h_sub)
                    a_int = self.W_a_int[l_idx - 1](int_node)
                    a = self.W_bmm[l_idx - 1](a_sub * a_int.index_select(0, batch_sub))
                    attn = scatter_softmax(a.view(-1), batch_sub).view(-1, 1)
                    m_main_to_int = scatter_sum(attn * self.W_main[l_idx - 1](h_sub), batch_sub, dim=0)
                    main_to_int = self.W_main_to_int[l_idx - 1](m_main_to_int)

                    int_self = self.W_int[l_idx - 1](int_node)
                    z_int = torch.sigmoid(self.W_zi1[l_idx - 1](int_self) + self.W_zi2[l_idx - 1](main_to_int))
                    hidden_int = (1 - z_int) * int_self + z_int * main_to_int
                    int_node = self.GRU_int(hidden_int, int_node)

                    int_to_main = self.W_int_to_main[l_idx - 1](int_node)

                    h, x = layer(h, h_sub, x, x_sub, edge_type, edge_type_sub, edge_index, edge_index_sub, mask_ligand, perm_sub, intnode=int_to_main, batch_sub=batch_sub, e_w=e_w, e_w_sub=None, fix_x=fix_x)

                subcomplex_pad = torch.zeros_like(h).to(h.device)
                subcomplex_pad[perm_sub] = h_sub

                subcomplex_pad = self.scmlp[l_idx](subcomplex_pad)

            all_x.append(x)
            all_h.append(h)

        outputs = {'x': x, 'h': h}
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})
        return outputs
