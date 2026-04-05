from .claude_code import ClaudeCodeParser
from .omp import OmpParser

PARSERS = {
    "claude_code": ClaudeCodeParser,
    "omp": OmpParser,
}
