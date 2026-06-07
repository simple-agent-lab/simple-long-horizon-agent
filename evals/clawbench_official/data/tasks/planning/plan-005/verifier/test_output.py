"""Verifier for plan-005: System Architecture Document."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def workspace(request):
    return Path(request.config.getoption("--workspace"))


@pytest.fixture
def architecture(workspace):
    path = workspace / "architecture.json"
    assert path.exists(), "architecture.json not found in workspace"
    return json.loads(path.read_text())


@pytest.fixture
def components(architecture):
    assert "components" in architecture, "Missing 'components' section"
    return architecture["components"]


@pytest.fixture
def data_flow(architecture):
    assert "data_flow" in architecture, "Missing 'data_flow' section"
    return architecture["data_flow"]


@pytest.fixture
def deployment(architecture):
    assert "deployment" in architecture, "Missing 'deployment' section"
    return architecture["deployment"]


@pytest.fixture
def tech_stack(architecture):
    assert "tech_stack" in architecture, "Missing 'tech_stack' section"
    return architecture["tech_stack"]


@pytest.fixture
def scaling_strategy(architecture):
    assert "scaling_strategy" in architecture, "Missing 'scaling_strategy' section"
    return architecture["scaling_strategy"]


# ── Core: Top-level structure (weight 3) ─────────────────────────────────────


@pytest.mark.weight(3)
def test_architecture_file_exists(workspace):
    assert (workspace / "architecture.json").exists(), "architecture.json not found"


@pytest.mark.weight(3)
def test_has_components_section(architecture):
    assert "components" in architecture, "Missing 'components' section"
    assert isinstance(architecture["components"], list), "'components' must be an array"


@pytest.mark.weight(3)
def test_has_data_flow_section(architecture):
    assert "data_flow" in architecture, "Missing 'data_flow' section"
    assert isinstance(architecture["data_flow"], list), "'data_flow' must be an array"


@pytest.mark.weight(3)
def test_has_deployment_section(architecture):
    assert "deployment" in architecture, "Missing 'deployment' section"
    assert isinstance(architecture["deployment"], dict), "'deployment' must be an object"


@pytest.mark.weight(3)
def test_has_tech_stack_section(architecture):
    assert "tech_stack" in architecture, "Missing 'tech_stack' section"
    assert isinstance(architecture["tech_stack"], dict), "'tech_stack' must be an object"


@pytest.mark.weight(3)
def test_has_scaling_strategy_section(architecture):
    assert "scaling_strategy" in architecture, "Missing 'scaling_strategy' section"


# ── Core: Components (weight 3) ──────────────────────────────────────────────


@pytest.mark.weight(3)
def test_components_minimum_count(components):
    assert len(components) >= 5, f"Need at least 5 components, found {len(components)}"


@pytest.mark.weight(3)
def test_components_have_name(components):
    for i, c in enumerate(components):
        assert "name" in c and isinstance(c["name"], str) and len(c["name"]) > 0, (
            f"Component {i} missing or empty 'name'"
        )


@pytest.mark.weight(3)
def test_components_have_responsibility(components):
    for i, c in enumerate(components):
        assert "responsibility" in c and isinstance(c["responsibility"], str), (
            f"Component {i} missing 'responsibility'"
        )
        assert len(c["responsibility"]) > 10, (
            f"Component {i}: 'responsibility' should be a meaningful description"
        )


@pytest.mark.weight(3)
def test_components_have_interfaces(components):
    for i, c in enumerate(components):
        assert "interfaces" in c and isinstance(c["interfaces"], list), (
            f"Component {i} missing 'interfaces' array"
        )
        assert len(c["interfaces"]) >= 1, (
            f"Component {i}: must expose at least 1 interface"
        )


# ── Core: Data Flow (weight 3) ──────────────────────────────────────────────


@pytest.mark.weight(3)
def test_data_flow_minimum_count(data_flow):
    assert len(data_flow) >= 4, f"Need at least 4 data flow connections, found {len(data_flow)}"


@pytest.mark.weight(3)
def test_data_flow_has_from(data_flow):
    for i, df in enumerate(data_flow):
        assert "from" in df and isinstance(df["from"], str), (
            f"Data flow {i} missing 'from'"
        )


@pytest.mark.weight(3)
def test_data_flow_has_to(data_flow):
    for i, df in enumerate(data_flow):
        assert "to" in df and isinstance(df["to"], str), (
            f"Data flow {i} missing 'to'"
        )


@pytest.mark.weight(3)
def test_data_flow_has_protocol(data_flow):
    for i, df in enumerate(data_flow):
        assert "protocol" in df and isinstance(df["protocol"], str), (
            f"Data flow {i} missing 'protocol'"
        )


@pytest.mark.weight(3)
def test_data_flow_has_description(data_flow):
    for i, df in enumerate(data_flow):
        assert "description" in df and isinstance(df["description"], str), (
            f"Data flow {i} missing 'description'"
        )


# ── Core: Deployment (weight 3) ──────────────────────────────────────────────


@pytest.mark.weight(3)
def test_deployment_has_environments(deployment):
    assert "environments" in deployment, "Deployment missing 'environments'"
    assert isinstance(deployment["environments"], list), "'environments' must be an array"
    assert len(deployment["environments"]) >= 3, (
        "Need at least 3 environments (development, staging, production)"
    )


@pytest.mark.weight(3)
def test_environments_have_name_and_purpose(deployment):
    for i, env in enumerate(deployment["environments"]):
        assert "name" in env and isinstance(env["name"], str), (
            f"Environment {i} missing 'name'"
        )
        assert "purpose" in env and isinstance(env["purpose"], str), (
            f"Environment {i} missing 'purpose'"
        )


# ── Core: Tech Stack (weight 3) ──────────────────────────────────────────────


@pytest.mark.weight(3)
def test_tech_stack_has_frontend(tech_stack):
    assert "frontend" in tech_stack and isinstance(tech_stack["frontend"], str), (
        "Tech stack missing 'frontend'"
    )
    assert len(tech_stack["frontend"]) > 3, "frontend must be a meaningful technology name"


@pytest.mark.weight(3)
def test_tech_stack_has_backend(tech_stack):
    assert "backend" in tech_stack and isinstance(tech_stack["backend"], str), (
        "Tech stack missing 'backend'"
    )
    assert len(tech_stack["backend"]) > 3, "backend must be a meaningful technology name"


@pytest.mark.weight(3)
def test_tech_stack_has_database(tech_stack):
    assert "database" in tech_stack and isinstance(tech_stack["database"], str), (
        "Tech stack missing 'database'"
    )
    assert len(tech_stack["database"]) > 3, "database must be a meaningful technology name"


@pytest.mark.weight(3)
def test_tech_stack_has_infrastructure(tech_stack):
    assert "infrastructure" in tech_stack and isinstance(tech_stack["infrastructure"], str), (
        "Tech stack missing 'infrastructure'"
    )
    assert len(tech_stack["infrastructure"]) > 3, "infrastructure must be a meaningful description"


# ── Bonus checks (weight 1) ──────────────────────────────────────────────────


@pytest.mark.weight(1)
def test_deployment_is_containerized(deployment):
    assert "containerized" in deployment, "Deployment should specify 'containerized'"
    assert deployment["containerized"] is True, "Modern architecture should be containerized"


@pytest.mark.weight(1)
def test_deployment_has_orchestration(deployment):
    assert "orchestration" in deployment and isinstance(deployment["orchestration"], str), (
        "Deployment should specify 'orchestration' tool"
    )


@pytest.mark.weight(1)
def test_scaling_has_approach(scaling_strategy):
    assert "approach" in scaling_strategy and isinstance(scaling_strategy["approach"], str), (
        "Scaling strategy should have an 'approach' description"
    )
    assert len(scaling_strategy["approach"]) > 20, "approach should be a detailed description"


@pytest.mark.weight(1)
def test_scaling_has_component_strategies(scaling_strategy):
    assert "components" in scaling_strategy and isinstance(scaling_strategy["components"], list), (
        "Scaling strategy should have 'components' array"
    )
    assert len(scaling_strategy["components"]) >= 2, (
        "Should have scaling strategies for at least 2 components"
    )


@pytest.mark.weight(1)
def test_component_names_unique(components):
    names = [c["name"] for c in components]
    assert len(names) == len(set(names)), "Component names must be unique"


@pytest.mark.weight(1)
def test_has_websocket_in_interfaces(components):
    all_interfaces = []
    for c in components:
        all_interfaces.extend(i.lower() for i in c.get("interfaces", []))
    combined = " ".join(all_interfaces)
    assert "websocket" in combined, (
        "A real-time chat architecture should have WebSocket in component interfaces"
    )
