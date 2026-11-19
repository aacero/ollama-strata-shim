import unittest
import json
from shim import (
    translate_options_to_openai,
    ollama_request_to_openai,
    openai_tool_calls_to_ollama,
)

class TestShimTranslations(unittest.TestCase):
    def test_translate_options(self):
        opts = {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 2048,
            "stop": ["<|im_end|>"],
            "seed": 42,
        }
        oa_opts = translate_options_to_openai(opts)
        self.assertEqual(oa_opts["temperature"], 0.7)
        self.assertEqual(oa_opts["top_p"], 0.9)
        self.assertEqual(oa_opts["max_tokens"], 2048)
        self.assertEqual(oa_opts["stop"], ["<|im_end|>"])
        self.assertEqual(oa_opts["seed"], 42)

    def test_ollama_request_tools_passthrough(self):
        payload = {
            "model": "qwen2.5-coder:32b",
            "messages": [{"role": "user", "content": "List files"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }
                }
            }],
            "stream": True,
        }
        oa = ollama_request_to_openai(payload)
        self.assertEqual(len(oa["tools"]), 1)
        self.assertEqual(oa["tools"][0]["function"]["name"], "list_directory")
        self.assertEqual(len(oa["messages"]), 1)
        self.assertEqual(oa["messages"][0]["role"], "user")

    def test_multi_turn_tool_history(self):
        payload = {
            "model": "qwen2.5-coder:32b",
            "messages": [
                {"role": "user", "content": "List files"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "list_directory",
                                "arguments": {"path": "."}
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_name": "list_directory",
                    "content": "file1.txt\nfile2.txt"
                }
            ],
            "stream": False,
        }
        oa = ollama_request_to_openai(payload)
        self.assertEqual(len(oa["messages"]), 3)

        # Check assistant message
        asst = oa["messages"][1]
        self.assertEqual(asst["role"], "assistant")
        self.assertIsNone(asst["content"])
        self.assertEqual(len(asst["tool_calls"]), 1)
        tc_id = asst["tool_calls"][0]["id"]
        self.assertTrue(tc_id.startswith("call_"))
        self.assertEqual(asst["tool_calls"][0]["type"], "function")
        self.assertEqual(asst["tool_calls"][0]["function"]["name"], "list_directory")
        self.assertEqual(json.loads(asst["tool_calls"][0]["function"]["arguments"]), {"path": "."})

        # Check tool result message
        tool_msg = oa["messages"][2]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["tool_call_id"], tc_id)
        self.assertEqual(tool_msg["content"], "file1.txt\nfile2.txt")

    def test_parallel_tool_calls_correlation(self):
        payload = {
            "model": "qwen2.5-coder:32b",
            "messages": [
                {"role": "user", "content": "Read two files"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "read_file", "arguments": {"path": "a.txt"}}},
                        {"function": {"name": "read_file", "arguments": {"path": "b.txt"}}},
                    ]
                },
                {"role": "tool", "content": "content of a"},
                {"role": "tool", "content": "content of b"},
            ]
        }
        oa = ollama_request_to_openai(payload)
        self.assertEqual(len(oa["messages"]), 4)
        id_a = oa["messages"][1]["tool_calls"][0]["id"]
        id_b = oa["messages"][1]["tool_calls"][1]["id"]
        self.assertNotEqual(id_a, id_b)
        self.assertEqual(oa["messages"][2]["tool_call_id"], id_a)
        self.assertEqual(oa["messages"][3]["tool_call_id"], id_b)

    def test_openai_tool_calls_to_ollama(self):
        oa_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "arguments": "{\"path\": \"/home\"}"
                }
            },
            {
                "id": "call_456",
                "type": "function",
                "function": {
                    "name": "invalid_json",
                    "arguments": "invalid{json"
                }
            }
        ]
        ollama_calls = openai_tool_calls_to_ollama(oa_calls)
        self.assertEqual(len(ollama_calls), 2)
        self.assertEqual(ollama_calls[0]["function"]["name"], "list_directory")
        self.assertEqual(ollama_calls[0]["function"]["arguments"], {"path": "/home"})
        self.assertEqual(ollama_calls[0]["id"], "call_123")
        self.assertEqual(ollama_calls[1]["function"]["arguments"], {"_raw": "invalid{json"})

if __name__ == "__main__":
    unittest.main()
