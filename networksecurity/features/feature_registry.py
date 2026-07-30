"""Feature registry: maps feature sets to dimension and names."""

FEATURE_REGISTRY: dict[str, dict] = {
    "afterimage": {
        "dim": 115,
        "description": "AfterImage incremental statistics — MAC/IP/transport in 5 time windows",
        "source": "networksecurity.engine.kitsune.afterimage.AfterImage",
    },
    "flow_statistical": {
        "dim": 9,
        "description": "Per-flow statistical features (duration, count, size, rate)",
        "source": "networksecurity.features.flow_extractor.FlowFeatures",
    },
    "lucid_per_packet": {
        "dim": 11,
        "description": "Per-packet features for LUCID CNN (packet_size, iat, protocol, flags, ports, direction, payload, header, window, ttl)",
        "source": "networksecurity.engine.lucid.dataset_parser.LucidDatasetParser",
    },
}


def get_feature_dim(name: str) -> int:
    return FEATURE_REGISTRY[name]["dim"]


def list_features() -> list[str]:
    return list(FEATURE_REGISTRY)
