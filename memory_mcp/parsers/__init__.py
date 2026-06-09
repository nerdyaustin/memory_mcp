from .claude_code import ClaudeCodeParser
from .codex import CodexParser
from .gemini import GeminiParser
from .lmstudio import LmStudioParser
from .lmstudio_api import LmStudioApiParser
from .omp import OmpParser

PARSERS = {
    "claude_code": ClaudeCodeParser,
    "codex": CodexParser,
    "gemini": GeminiParser,
    "lmstudio": LmStudioParser,
    "lmstudio_api": LmStudioApiParser,
    "omp": OmpParser,
}
