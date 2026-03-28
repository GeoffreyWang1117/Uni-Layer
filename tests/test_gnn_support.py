"""
Tests for GNN (Graph Neural Network) support (v0.4.0 Feature 5).
"""

import pytest
import torch
import torch.nn as nn

# Skip all tests if PyG is not installed
torch_geometric = pytest.importorskip("torch_geometric")

from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool


class SimpleGCN(nn.Module):
    """Simple GCN model for testing."""

    def __init__(self, in_dim=16, hidden_dim=32, out_dim=5, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        return self.head(x)


class SimpleGAT(nn.Module):
    """Simple GAT model for testing."""

    def __init__(self, in_dim=16, hidden_dim=32, out_dim=5):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=2, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch=None):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        return self.head(x)


def make_graph_batch(num_graphs=8, num_nodes=20, num_edges=40, node_dim=16, num_classes=5):
    """Create a batch of random graphs."""
    graphs = []
    for _ in range(num_graphs):
        x = torch.randn(num_nodes, node_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        y = torch.randint(0, num_classes, (1,))
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    return Batch.from_data_list(graphs)


@pytest.fixture
def gcn_model():
    return SimpleGCN(in_dim=16, hidden_dim=32, out_dim=5, num_layers=3)


@pytest.fixture
def gat_model():
    return SimpleGAT(in_dim=16, hidden_dim=32, out_dim=5)


@pytest.fixture
def graph_batch():
    return make_graph_batch()


@pytest.fixture
def graph_dataloader():
    """Simulate a DataLoader that yields (batch, targets) tuples."""
    batches = []
    for _ in range(5):
        batch = make_graph_batch()
        targets = batch.y.squeeze()
        batches.append((batch, targets))
    return batches


class TestGNNLayerDetection:
    def test_identify_gcn_conv(self):
        from uni_layer.utils.layer_utils import identify_layer_type

        conv = GCNConv(16, 32)
        assert identify_layer_type(conv) == "gnn_conv"

    def test_identify_gat_conv(self):
        from uni_layer.utils.layer_utils import identify_layer_type

        conv = GATConv(16, 32)
        assert identify_layer_type(conv) == "gnn_attention"

    def test_identify_sage_conv(self):
        from uni_layer.utils.layer_utils import identify_layer_type

        conv = SAGEConv(16, 32)
        assert identify_layer_type(conv) == "gnn_sage"

    def test_get_model_layers_gcn(self, gcn_model):
        from uni_layer.utils.layer_utils import get_model_layers

        layers = get_model_layers(gcn_model)
        # Should find 3 GCNConv layers
        assert len(layers) == 3
        for name, module in layers.items():
            assert isinstance(module, GCNConv)

    def test_get_model_layers_gat(self, gat_model):
        from uni_layer.utils.layer_utils import get_model_layers

        layers = get_model_layers(gat_model)
        assert len(layers) == 2
        for name, module in layers.items():
            assert isinstance(module, GATConv)

    def test_architecture_family(self):
        from uni_layer.utils.layer_utils import get_architecture_family

        class GCNModel(nn.Module):
            pass

        # Current detection is name-based, so "gcn" in class name triggers it
        model = GCNModel()
        assert get_architecture_family(model) == "gcn_family"


class TestGNNModelForward:
    def test_pyg_data_forward(self, gcn_model, graph_batch):
        """model_forward should handle PyG Data/Batch objects."""
        from uni_layer.utils.model_adapter import model_forward

        output = model_forward(gcn_model, graph_batch)
        assert isinstance(output, torch.Tensor)
        assert output.shape[-1] == 5  # num_classes

    def test_pyg_data_detection(self, graph_batch):
        from uni_layer.utils.model_adapter import _is_pyg_data

        assert _is_pyg_data(graph_batch)
        assert not _is_pyg_data(torch.randn(10, 16))

    def test_gat_forward(self, gat_model, graph_batch):
        from uni_layer.utils.model_adapter import model_forward

        output = model_forward(gat_model, graph_batch)
        assert isinstance(output, torch.Tensor)


class TestGNNMetrics:
    def test_effective_rank_gcn(self, gcn_model, graph_dataloader):
        """EffectiveRank should work on GCN conv layers."""
        from uni_layer.metrics import EffectiveRank
        from uni_layer.utils.layer_utils import get_model_layers

        layers = get_model_layers(gcn_model)
        layer_name = list(layers.keys())[0]
        layer = layers[layer_name]

        m = EffectiveRank(num_batches=2)
        result = m.compute(
            model=gcn_model,
            layer=layer,
            layer_name=layer_name,
            layer_idx=0,
            data_loader=graph_dataloader,
            device="cpu",
        )

        assert "effective_rank" in result
        assert isinstance(result["effective_rank"], float)
        assert result["effective_rank"] > 0

    def test_activation_entropy_gcn(self, gcn_model, graph_dataloader):
        """ActivationEntropy should work on GCN conv layers."""
        from uni_layer.metrics import ActivationEntropy
        from uni_layer.utils.layer_utils import get_model_layers

        layers = get_model_layers(gcn_model)
        layer_name = list(layers.keys())[0]
        layer = layers[layer_name]

        m = ActivationEntropy(num_batches=2)
        result = m.compute(
            model=gcn_model,
            layer=layer,
            layer_name=layer_name,
            layer_idx=0,
            data_loader=graph_dataloader,
            device="cpu",
        )

        assert "activation_entropy" in result
        assert isinstance(result["activation_entropy"], float)

    def test_block_influence_gat(self, gat_model, graph_dataloader):
        """BlockInfluence should work on GAT conv layers."""
        from uni_layer.metrics import BlockInfluence
        from uni_layer.utils.layer_utils import get_model_layers

        layers = get_model_layers(gat_model)
        layer_name = list(layers.keys())[0]
        layer = layers[layer_name]

        m = BlockInfluence(num_batches=2)
        result = m.compute(
            model=gat_model,
            layer=layer,
            layer_name=layer_name,
            layer_idx=0,
            data_loader=graph_dataloader,
            device="cpu",
        )

        assert "block_influence" in result
        assert isinstance(result["block_influence"], float)
