"""Human-readable names for the review clusters used in the dashboard."""

CLUSTER_CATEGORIES = {
    0: "Computers, Storage & Peripherals",
    1: "Audio, Headphones & Connected Devices",
    2: "Cases, Covers & Tablet Accessories",
    3: "Cameras, Video & Smart Home",
    4: "Cables, Chargers & Power Accessories",
    5: "General Electronics & Everyday Accessories",
}


def get_cluster_category(cluster_id: int) -> str:
    """Return a display name while preserving the numeric cluster ID in data."""
    return CLUSTER_CATEGORIES.get(cluster_id, f"Cluster {cluster_id}")
