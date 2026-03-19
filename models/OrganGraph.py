import json
import os
from typing import List, Tuple, Dict, Set
from collections import defaultdict

class OrganGraph:
    """
    Organ-Centric Graph Module
    
    Loads and manages the organ-centric graph structure from organ_graph.json.
    Provides interfaces to retrieve knowledge triplets for feature transformation.
    """
    
    def __init__(self, graph_path: str):
        """
        Initialize the organ-centric graph.
        
        Args:
            graph_path: Path to organ_graph.json file
        """
        with open(graph_path, 'r') as f:
            self.graph_data = json.load(f)
        
        # Extract all unique entities (lab tests, organs, imaging abnormalities)
        self.lab_tests = []
        self.imaging_abnormalities = []
        self.organs = set()
        
        # Build mappings: lab_test/abnormality -> list of (organ, relation) tuples
        self.lab_to_organs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.abn_to_organs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        
        # Build reverse mappings: organ -> list of (lab_test/abnormality, relation) tuples
        self.org_to_labs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.org_to_abns: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        
        # Process graph data
        self._build_graph()
        
        # Create sorted lists for consistent indexing
        self.lab_test_list = sorted(self.lab_tests)
        self.abnormality_list = sorted(self.imaging_abnormalities)
        self.organ_list = sorted(list(self.organs))
        
        # Create index mappings
        self.lab_to_idx = {lab: idx for idx, lab in enumerate(self.lab_test_list)}
        self.abn_to_idx = {abn: idx for idx, abn in enumerate(self.abnormality_list)}
        self.org_to_idx = {org: idx for idx, org in enumerate(self.organ_list)}
        
    def _build_graph(self):
        """Build graph structure from JSON data."""
        # Known imaging abnormality names (from dataloader)
        known_abnormalities = {
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
            'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax'
        }
        
        for entry in self.graph_data:
            entity_id = entry['id']
            relations = entry['relation']
            
            # Determine if this is a lab test or imaging abnormality
            is_abnormality = entity_id in known_abnormalities
            
            if is_abnormality:
                self.imaging_abnormalities.append(entity_id)
            else:
                self.lab_tests.append(entity_id)
            
            # Process each relation triplet
            for triplet in relations:
                if len(triplet) != 3:
                    continue
                
                entity1, relation, entity2 = triplet
                
                # entity1 should match entity_id
                if entity1 != entity_id:
                    continue
                
                # entity2 is the organ
                organ = entity2
                self.organs.add(organ)
                
                # Build mappings
                if is_abnormality:
                    self.abn_to_organs[entity_id].append((organ, relation))
                    self.org_to_abns[organ].append((entity_id, relation))
                else:
                    self.lab_to_organs[entity_id].append((organ, relation))
                    self.org_to_labs[organ].append((entity_id, relation))
    
    def get_lab_triplets(self, lab_test_name: str) -> List[Tuple[str, str, str]]:
        """
        Get knowledge triplets for a laboratory test.
        
        Args:
            lab_test_name: Name of the laboratory test
        
        Returns:
            List of (lab_test, relation, organ) triplets
        """
        triplets = []
        if lab_test_name in self.lab_to_organs:
            for organ, relation in self.lab_to_organs[lab_test_name]:
                triplets.append((lab_test_name, relation, organ))
        return triplets
    
    def get_abnormality_triplets(self, abnormality_name: str) -> List[Tuple[str, str, str]]:
        """
        Get knowledge triplets for an imaging abnormality.
        
        Args:
            abnormality_name: Name of the imaging abnormality
        
        Returns:
            List of (abnormality, relation, organ) triplets
        """
        triplets = []
        if abnormality_name in self.abn_to_organs:
            for organ, relation in self.abn_to_organs[abnormality_name]:
                triplets.append((abnormality_name, relation, organ))
        return triplets
    
    def get_organs_for_lab(self, lab_test_name: str) -> List[str]:
        """Get list of organs connected to a lab test."""
        return [organ for organ, _ in self.lab_to_organs.get(lab_test_name, [])]
    
    def get_organs_for_abnormality(self, abnormality_name: str) -> List[str]:
        """Get list of organs connected to an imaging abnormality."""
        return [organ for organ, _ in self.abn_to_organs.get(abnormality_name, [])]
    
    def get_labs_for_organ(self, organ_name: str) -> List[str]:
        """Get list of lab tests connected to an organ."""
        return [lab for lab, _ in self.org_to_labs.get(organ_name, [])]
    
    def get_abnormalities_for_organ(self, organ_name: str) -> List[str]:
        """Get list of imaging abnormalities connected to an organ."""
        return [abn for abn, _ in self.org_to_abns.get(organ_name, [])]
    
    def get_num_lab_tests(self) -> int:
        """Get number of lab tests in the graph."""
        return len(self.lab_test_list)
    
    def get_num_abnormalities(self) -> int:
        """Get number of imaging abnormalities in the graph."""
        return len(self.abnormality_list)
    
    def get_num_organs(self) -> int:
        """Get number of organs in the graph."""
        return len(self.organ_list)
    
    def get_lab_test_index(self, lab_test_name: str) -> int:
        """Get index of a lab test in the sorted list."""
        return self.lab_to_idx.get(lab_test_name, -1)
    
    def get_abnormality_index(self, abnormality_name: str) -> int:
        """Get index of an imaging abnormality in the sorted list."""
        return self.abn_to_idx.get(abnormality_name, -1)
    
    def get_organ_index(self, organ_name: str) -> int:
        """Get index of an organ in the sorted list."""
        return self.org_to_idx.get(organ_name, -1)
    
    def get_all_lab_tests(self) -> List[str]:
        """Get all lab test names."""
        return self.lab_test_list.copy()
    
    def get_all_abnormalities(self) -> List[str]:
        """Get all imaging abnormality names."""
        return self.abnormality_list.copy()
    
    def get_all_organs(self) -> List[str]:
        """Get all organ names."""
        return self.organ_list.copy()

