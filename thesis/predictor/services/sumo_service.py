"""SUMO service for SUMO network operations."""

from pathlib import Path

import sumolib

from thesis.common.config import NETWORK_BASE_FILENAME


class SumoService:
    """SUMO service for SUMO network operations."""

    def __init__(self, common_dir: Path) -> None:
        self._network: sumolib.net.Net = sumolib.net.readNet(common_dir / NETWORK_BASE_FILENAME)

    def _lonlat_to_xy(self, longitude: float, latitude: float) -> tuple[float, float]:
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

    def _calculate_shortest_path(
        self, source_x: float, source_y: float, destination_x: float, destination_y: float
    ) -> tuple[list[sumolib.net.edge.Edge], float] | tuple[None, None]:
        """
        Calculate the shortest path edges and distance between two points.

        Args:
            source_x (float): Source x coordinate.
            source_y (float): Source y coordinate.
            destination_x (float): Destination x coordinate.
            destination_y (float): Destination y coordinate.

        Returns:
            tuple[list[sumolib.net.edge.Edge], float] | tuple[None, None]: Tuple of edge list and distance or None if no path found.
        """
        source_neighboring_edges = self._network.getNeighboringEdges(source_x, source_y, 500)
        destination_neighboring_edges = self._network.getNeighboringEdges(destination_x, destination_y, 500)

        if not source_neighboring_edges or not destination_neighboring_edges:
            return None, None

        source_edge, _ = min(source_neighboring_edges, key=lambda t: t[1])
        destination_edge, _ = min(destination_neighboring_edges, key=lambda t: t[1])

        edge_list, distance = self._network.getShortestPath(source_edge, destination_edge)

        return edge_list, distance

    def _find_closest_point_on_segment(
        self, point: tuple[float, float], segment_start: tuple[float, float], segment_end: tuple[float, float]
    ) -> tuple[float, float]:
        """
        Find the closest point on a line segment to a given point.

        Args:
            point (tuple[float, float]): The point (lat, lon).
            segment_start (tuple[float, float]): Segment start (lat, lon).
            segment_end (tuple[float, float]): Segment end (lat, lon).

        Returns:
            tuple[float, float]: The closest point on the segment.
        """
        px, py = point
        ax, ay = segment_start
        bx, by = segment_end

        dx = bx - ax
        dy = by - ay

        if dx == 0 and dy == 0:
            return segment_start

        t_numerator = (px - ax) * dx + (py - ay) * dy
        t_denominator = dx * dx + dy * dy
        t = t_numerator / t_denominator
        t_clamped = max(0, min(1, t))

        closest_x = ax + t_clamped * dx
        closest_y = ay + t_clamped * dy

        return (closest_x, closest_y)

    def _find_closest_segment(
        self, point: tuple[float, float], route: list[tuple[float, float]]
    ) -> tuple[int, tuple[float, float]]:
        """
        Find which segment in the route is closest to the given point.

        Args:
            point (tuple[float, float]): The point to find closest segment for.
            route (list[tuple[float, float]]): The route polyline.

        Returns:
            tuple[int, tuple[float, float]]: Index of segment start and the projected point on that segment.
        """
        min_distance = float("inf")
        closest_idx = 0
        closest_projection = route[0]

        for i in range(len(route) - 1):
            projection = self._find_closest_point_on_segment(point, route[i], route[i + 1])
            distance = ((point[0] - projection[0]) ** 2 + (point[1] - projection[1]) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
                closest_projection = projection

        return closest_idx, closest_projection

    def _trim_route_to_points(
        self,
        route: list[tuple[float, float]],
        source_lat: float,
        source_lon: float,
        dest_lat: float,
        dest_lon: float,
    ) -> list[tuple[float, float]]:
        """
        Trim the route to start at source point and end at destination point.

        Args:
            route (list[tuple[float, float]]): Full route polyline.
            source_lat (float): Source latitude.
            source_lon (float): Source longitude.
            dest_lat (float): Destination latitude.
            dest_lon (float): Destination longitude.

        Returns:
            list[tuple[float, float]]: Trimmed route.
        """
        if len(route) < 2:
            return route

        source_point = (source_lat, source_lon)
        dest_point = (dest_lat, dest_lon)

        source_idx, closest_start = self._find_closest_segment(source_point, route)
        dest_idx, closest_end = self._find_closest_segment(dest_point, route)

        trimmed = [closest_start]
        if dest_idx > source_idx:
            trimmed.extend(route[source_idx + 1 : dest_idx + 1])
        trimmed.append(closest_end)

        return trimmed

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

    def compute_trip_features(
        self, source_latitude: float, source_longitude: float, destination_latitude: float, destination_longitude: float
    ) -> dict[str, float | list[str] | list[tuple[float, float]]]:
        """
        Compute all trip features including route, distance, edges, and coordinates.

        Args:
            source_latitude (float): Source latitude.
            source_longitude (float): Source longitude.
            destination_latitude (float): Destination latitude.
            destination_longitude (float): Destination longitude.

        Returns:
            dict[str, float | list[str] | list[tuple[float, float]]]: Dictionary with all computed features.
        """
        source_x, source_y = self._lonlat_to_xy(source_longitude, source_latitude)
        destination_x, destination_y = self._lonlat_to_xy(destination_longitude, destination_latitude)
        edge_list, distance = self._calculate_shortest_path(source_x, source_y, destination_x, destination_y)

        if edge_list is None or distance is None:
            distance = ((destination_x - source_x) ** 2 + (destination_y - source_y) ** 2) ** 0.5
            edges = []
            route = [(source_latitude, source_longitude), (destination_latitude, destination_longitude)]
        else:
            edges = [edge.getID() for edge in edge_list]
            path: list[tuple[float, float]] = []

            for i, edge_id in enumerate(edges):
                segment = self._edge_lonlat_shape(edge_id)
                if i > 0 and path and segment:
                    if path[-1] == segment[0]:
                        path.extend(segment[1:])
                    else:
                        path.extend(segment)
                else:
                    path.extend(segment)

            route = self._trim_route_to_points(
                path, source_latitude, source_longitude, destination_latitude, destination_longitude
            )

        return {
            "source_x": source_x,
            "source_y": source_y,
            "destination_x": destination_x,
            "destination_y": destination_y,
            "distance": distance,
            "edges": edges,
            "route": route,
        }

    def clear(self) -> None:
        """Clear the SUMO service."""
        pass
