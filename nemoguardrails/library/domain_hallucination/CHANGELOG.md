# Changelog

All notable changes to the Domain Hallucination Guard library are documented in this file.

## [0.1.0] - 2026-06-01

### Added

#### Core Features
- Entity extraction (URLs, domains, GitHub repos)
- Multi-level verification (DNS, HTTP, GitHub API)
- Knowledge base management (seed KB + external KB)
- Risk scoring with issue aggregation
- Score recalibration based on verification evidence
- Decision making with configurable thresholds
- Answer enforcement (block, refine, warn, pass)

#### Modules
- `extractors.py` - URL, domain, and GitHub extraction
- `verification.py` - DNS, HTTP, and GitHub verification
- `checkers.py` - Issue aggregation and detection
- `scoring.py` - Risk scoring and recalibration
- `decision.py` - Decision making and enforcement
- `kb.py` - Knowledge base management
- `semantic.py` - Semantic relevance and advanced verification
- `actions.py` - Main NeMo action
- `config.py` - Configuration management
- `schemas.py` - Data structures
- `utils.py` - Utility functions
- `examples.py` - Usage examples

#### Configuration
- JSON-based configuration
- Environment variable support
- Multiple verification levels (none, dns, http, full)
- Configurable scoring thresholds
- Customizable enforcement messages

#### Knowledge Base
- Seed KB with trusted domains and repos
- External KB root support
- Blacklist management
- Metadata support for entities

#### Integrations
- NeMo Guardrails adapter
- FastAPI integration example
- LangChain integration example
- Docker deployment support

#### Documentation
- Comprehensive README
- Architecture design document
- Integration guide
- API examples
- Inline docstrings

#### Testing
- Unit tests for extractors
- Unit tests for verification
- Unit tests for scoring
- Unit tests for KB
- Example test cases

### Initial Implementation
- All core components functional
- First stable release
- Ready for production use
