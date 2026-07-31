from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simple_long_horizon_agent.mcp.config_file import (
    load_mcp_server_configs,
    mcp_server_configs_from_json,
)


class McpConfigFileTest(unittest.TestCase):
    def test_loads_stdio_server_configs_from_json_file(self) -> None:
        payload = {
            "servers": [
                {
                    "name": "workspace",
                    "transport": "stdio",
                    "command": "python",
                    "args": ["-m", "server"],
                    "cwd": "/testbed",
                    "env": {"TOKEN": "value"},
                    "init_timeout": 10,
                    "call_timeout": 20,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mcp.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            configs = load_mcp_server_configs(path)

        self.assertEqual(len(configs), 1)
        config = configs[0]
        self.assertEqual(config.name, "workspace")
        self.assertEqual(config.transport, "stdio")
        self.assertEqual(config.command, "python")
        self.assertEqual(config.args, ("-m", "server"))
        self.assertEqual(config.cwd, "/testbed")
        self.assertEqual(config.env, {"TOKEN": "value"})
        self.assertEqual(config.init_timeout, 10)
        self.assertEqual(config.call_timeout, 20)

    def test_loads_http_server_config(self) -> None:
        configs = mcp_server_configs_from_json(
            json.dumps(
                {
                    "servers": [
                        {
                            "name": "remote",
                            "transport": "http",
                            "url": "https://example.invalid/mcp",
                            "headers": {"Authorization": "Bearer token"},
                        }
                    ]
                }
            ),
            source="inline",
        )

        self.assertEqual(configs[0].name, "remote")
        self.assertEqual(configs[0].transport, "http")
        self.assertEqual(configs[0].url, "https://example.invalid/mcp")
        self.assertEqual(configs[0].headers, {"Authorization": "Bearer token"})

    def test_rejects_unknown_server_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown field"):
            mcp_server_configs_from_json(
                json.dumps(
                    {
                        "servers": [
                            {
                                "name": "workspace",
                                "transport": "stdio",
                                "command": "python",
                                "unexpected": True,
                            }
                        ]
                    }
                ),
                source="inline",
            )

    def test_requires_servers_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "servers"):
            mcp_server_configs_from_json("{}", source="inline")
