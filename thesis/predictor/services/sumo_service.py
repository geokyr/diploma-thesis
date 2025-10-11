"""SUMO service for SUMO network operations."""

from functools import lru_cache
from pathlib import Path

import sumolib

from thesis.common.config import NETWORK_BASE_FILENAME


class SumoService:
    """SUMO service for SUMO network operations."""

    def __init__(self, common_dir: Path) -> None:
        self._network: sumolib.net.Net = sumolib.net.readNet(common_dir / NETWORK_BASE_FILENAME)

    def lonlat_to_xy(self, longitude: float, latitude: float) -> tuple[float, float]:
        """
        Convert longitude/latitude to x/y coordinates.

        Args:
            longitude (float): Longitude.
            latitude (float): Latitude.

        Returns:
            tuple[float, float]: x, y coordinates.
        """
        x, y = self._network.convertLonLat2XY(longitude, latitude)
        return x, y

    def _get_shortest_path(
        self, source_x: float, source_y: float, destination_x: float, destination_y: float
    ) -> tuple[list, float] | tuple[None, None]:
        """
        Get the shortest path edges and distance between two points.

        Args:
            source_x (float): Source x coordinate.
            source_y (float): Source y coordinate.
            destination_x (float): Destination x coordinate.
            destination_y (float): Destination y coordinate.

        Returns:
            tuple[list, float] | tuple[None, None]: Tuple of edges and distance or None if no path found.
        """
        source_neighboring_edges = self._network.getNeighboringEdges(source_x, source_y, 500)
        destination_neighboring_edges = self._network.getNeighboringEdges(destination_x, destination_y, 500)

        if not source_neighboring_edges or not destination_neighboring_edges:
            return None, None

        source_edge, _ = min(source_neighboring_edges, key=lambda t: t[1])
        destination_edge, _ = min(destination_neighboring_edges, key=lambda t: t[1])

        edges, distance = self._network.getShortestPath(source_edge, destination_edge)

        return edges, distance

    def calculate_trip_distance(
        self, source_x: float, source_y: float, destination_x: float, destination_y: float
    ) -> float:
        """
        Calculate the trip distance between two points using the SUMO network.

        Args:
            source_x (float): Source x coordinate.
            source_y (float): Source y coordinate.
            destination_x (float): Destination x coordinate.
            destination_y (float): Destination y coordinate.

        Returns:
            float: Trip distance in meters.
        """
        _, distance = self._get_shortest_path(source_x, source_y, destination_x, destination_y)

        if distance is None:
            return ((destination_x - source_x) ** 2 + (destination_y - source_y) ** 2) ** 0.5

        return distance

    def get_trip_edges(self, source_x: float, source_y: float, dest_x: float, dest_y: float) -> list[str]:
        """
        Get the list of edge IDs along the trip between two points.

        Args:
            source_x (float): Source x coordinate.
            source_y (float): Source y coordinate.
            dest_x (float): Destination x coordinate.
            dest_y (float): Destination y coordinate.

        Returns:
            list[str]: List of edge IDs along the trip.
        """
        edges, _ = self._get_shortest_path(source_x, source_y, dest_x, dest_y)

        if edges is None:
            return []

        return [edge.getID() for edge in edges]

    def clear(self) -> None:
        """Clear the SUMO service."""
        self._network = None

    @lru_cache(maxsize=2000)
    def _edge_lonlat_shape(self, edge_id: str) -> list[tuple[float, float]]:
        """
        Return this edge's shape as a list of (lat, lon) tuples.

        Args:
            edge_id (str): The ID of the edge.

        Returns:
            list[tuple[float, float]]: The shape of the edge as a list of (lat, lon) tuples.
        """
        edge: sumolib.net.edge.Edge = self._network.getEdge(edge_id)
        xy_shape: list[tuple[float, float]] = edge.getShape() or [
            edge.getFromNode().getCoord(),
            edge.getToNode().getCoord(),
        ]
        lonlat = [self._network.convertXY2LonLat(x, y) for (x, y) in xy_shape]
        return [(lat, lon) for (lon, lat) in lonlat]

    def route_lonlat_polyline(
        self, source_x: float, source_y: float, destination_x: float, destination_y: float
    ) -> list[tuple[float, float]]:
        """
        Compute shortest path and return a single polyline as a list of (lat, lon) tuples.

        Args:
            source_x (float): The x coordinate of the source point.
            source_y (float): The y coordinate of the source point.
            destination_x (float): The x coordinate of the destination point.
            destination_y (float): The y coordinate of the destination point.

        Returns:
            list[tuple[float, float]]: The polyline as a list of (lat, lon) tuples.
        """
        edges, _ = self._get_shortest_path(source_x, source_y, destination_x, destination_y)
        if not edges:
            source_longitude, source_latitude = self._network.convertXY2LonLat(source_x, source_y)
            destination_longitude, destination_latitude = self._network.convertXY2LonLat(destination_x, destination_y)
            return [(source_latitude, source_longitude), (destination_latitude, destination_longitude)]

        path: list[tuple[float, float]] = []
        for i, edge in enumerate(edges):
            segment = self._edge_lonlat_shape(edge.getID())
            if i > 0 and path and segment:
                if path[-1] == segment[0]:
                    path.extend(segment[1:])
                else:
                    path.extend(segment)
            else:
                path.extend(segment)

        return path
