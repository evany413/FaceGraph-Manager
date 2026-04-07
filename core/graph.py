import networkx as nx

from core.database import DatabaseManager


class GraphManager:
    """
    Derives folder-level edges from face connections and computes
    connected components for consolidation grouping.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    def build_folder_graph(self) -> nx.Graph:
        """
        Returns a NetworkX graph where nodes are folder_ids and edges
        represent at least one face connection between the two folders.
        """
        g = nx.Graph()
        for folder in self.db.get_all_folders():
            g.add_node(folder["folder_id"])

        connections = self.db.get_all_face_connections()
        identities = {i["cluster_id"]: i for i in self.db.get_all_identities()}

        for cid_a, cid_b in connections:
            id_a = identities.get(cid_a)
            id_b = identities.get(cid_b)
            if id_a and id_b and id_a["folder_id"] != id_b["folder_id"]:
                g.add_edge(id_a["folder_id"], id_b["folder_id"])

        return g

    def get_folder_edges(self) -> list[tuple[int, int]]:
        """Returns list of (folder_id_a, folder_id_b) derived group edges."""
        g = self.build_folder_graph()
        return list(g.edges())

    def get_consolidation_groups(self) -> list[list[int]]:
        """
        Returns connected components as lists of folder_ids.
        Each component = folders to be moved under one parent directory.
        Only components with 2+ folders are included.
        """
        g = self.build_folder_graph()
        components = []
        for component in nx.connected_components(g):
            if len(component) >= 2:
                components.append(sorted(component))
        return components
