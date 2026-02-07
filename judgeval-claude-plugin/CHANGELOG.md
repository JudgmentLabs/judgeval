# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-19

### Added
- Initial release of Judgeval Claude Code plugin
- `trace-claude-code` skill for automatic conversation tracing
- Session, turn, LLM, and tool span tracking
- Token usage metrics with cache support
- Multi-instance support with atomic state management
- OTLP HTTP/JSON export to Judgment Labs API

### Features
- Automatic tracing of Claude Code sessions
- Accurate duration tracking for LLM calls and tool invocations
- Parallel tool call support
- File locking for concurrent session safety
- Background API calls to prevent hook preemption
