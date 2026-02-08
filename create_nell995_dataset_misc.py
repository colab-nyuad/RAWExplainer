"""
Script to create NELL995 dataset in the PyTorch Geometric format.
"""

import os
import torch
import pandas as pd
from torch_geometric.data import Data
from collections import defaultdict

def load_triples(file_path):
    """Load triples from a text file."""
    triples = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                head, relation, tail = line.split('\t')
                triples.append((head, relation, tail))
    return triples

def create_entity_relation_mappings(triples):
    """Create entity and relation ID mappings."""
    entities = set()
    relations = set()
    
    for head, relation, tail in triples:
        entities.add(head)
        entities.add(tail)
        relations.add(relation)
    
    # Create mappings
    entity2id = {entity: idx for idx, entity in enumerate(sorted(entities))}
    relation2id = {relation: idx for idx, relation in enumerate(sorted(relations))}
    
    id2entity = {idx: entity for entity, idx in entity2id.items()}
    id2relation = {idx: relation for relation, idx in relation2id.items()}
    
    return entity2id, relation2id, id2entity, id2relation

def create_pyg_dataset(train_triples, valid_triples, test_triples, entity2id, relation2id):
    """Create PyTorch Geometric dataset."""
    
    def triples_to_edges(triples, entity2id, relation2id):
        """Convert triples to edge_index and edge_type tensors."""
        heads = []
        tails = []
        relations = []
        
        for head, relation, tail in triples:
            heads.append(entity2id[head])
            tails.append(entity2id[tail])
            relations.append(relation2id[relation])
        
        edge_index = torch.tensor([heads, tails], dtype=torch.long)
        edge_type = torch.tensor(relations, dtype=torch.long)
        
        return edge_index, edge_type
    
    # Convert all triples to edges (for the full graph)
    all_triples = train_triples + valid_triples + test_triples
    edge_index, edge_type = triples_to_edges(all_triples, entity2id, relation2id)
    
    # Create target edges for each split
    train_edge_index, train_edge_type = triples_to_edges(train_triples, entity2id, relation2id)
    valid_edge_index, valid_edge_type = triples_to_edges(valid_triples, entity2id, relation2id)
    test_edge_index, test_edge_type = triples_to_edges(test_triples, entity2id, relation2id)
    
    num_nodes = len(entity2id)
    
    # Create Data objects for each split
    train_data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        target_edge_index=train_edge_index,
        target_edge_type=train_edge_type,
        split='train',
        num_nodes=num_nodes
    )
    
    valid_data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        target_edge_index=valid_edge_index,
        target_edge_type=valid_edge_type,
        split='valid',
        num_nodes=num_nodes
    )
    
    test_data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        target_edge_index=test_edge_index,
        target_edge_type=test_edge_type,
        split='test',
        num_nodes=num_nodes
    )
    
    return [train_data, valid_data, test_data]

def main():
    # Paths
    base_dir = "/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG"
    nell995_dir = os.path.join(base_dir, "datasets/knowledge_graphs/nell995/raw")
    misc_dir = os.path.join(base_dir, "misc")
    
    # Load triples
    print("Loading NELL995 triples...")
    train_triples = load_triples(os.path.join(nell995_dir, "train.txt"))
    valid_triples = load_triples(os.path.join(nell995_dir, "valid.txt"))
    test_triples = load_triples(os.path.join(nell995_dir, "test.txt"))
    
    print(f"Loaded {len(train_triples)} train, {len(valid_triples)} valid, {len(test_triples)} test triples")
    
    # Create mappings
    print("Creating entity and relation mappings...")
    entity2id, relation2id, id2entity, id2relation = create_entity_relation_mappings(
        train_triples + valid_triples + test_triples
    )
    
    print(f"Found {len(entity2id)} entities and {len(relation2id)} relations")
    
    # Create PyG dataset
    print("Creating PyTorch Geometric dataset...")
    dataset = create_pyg_dataset(train_triples, valid_triples, test_triples, entity2id, relation2id)
    
    # Save files
    dataset_path = os.path.join(misc_dir, "nell995_dataset.pt")
    id2name_path = os.path.join(misc_dir, "nell995_id2name.pt")
    vocab_path = os.path.join(base_dir, "data/nell995_entity.txt")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
    
    print(f"Saving dataset to {dataset_path}")
    torch.save(dataset, dataset_path)
    
    print(f"Saving id2name mappings to {id2name_path}")
    torch.save((id2entity, id2relation), id2name_path)
    
    print(f"Saving vocabulary to {vocab_path}")
    with open(vocab_path, 'w') as f:
        for entity, entity_id in entity2id.items():
            f.write(f"{entity}\t{entity}\n")  # entity name as both key and value
    
    print("Done! NELL995 dataset files created successfully.")

if __name__ == "__main__":
    main()
