# Changelog

All notable changes to the Colang language and runtime will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0-beta.6] - 2025-01-16

### Added

* Add support for llama-3.2 models ([#877](https://github.com/NVIDIA/NeMo-Guardrails/pull/877)) by @schuellc-nvidia
* Add `it finished` utility flow in core.co library ([#913]<https://github.com/NVIDIA/NeMo-Guardrails/pull/913>) by @schuellc-nvidia

## [2.0-beta.5] - 2024-11-19

### Added

* Prompt template name to verbose logging ([#811](https://github.com/NVIDIA/NeMo-Guardrails/pull/811)) by @schuellc-nvidia
* New configuration setting to change UMIM event source id ([#823](https://github.com/NVIDIA/NeMo-Guardrails/pull/823)) by @sklinglernv
* New attention module to standard library ([#829](https://github.com/NVIDIA/NeMo-Guardrails/pull/829)) by @sklinglernv
* Passthrough mode support ([#779](https://github.com/NVIDIA/NeMo-Guardrails/pull/779)) by @Pouyanpi

### Fixed

* Activation of flows with default parameters ([#758](https://github.com/NVIDIA/NeMo-Guardrails/pull/758)) by @schuellc-nvidia
* ``pretty_str`` string formatting function ([#759](https://github.com/NVIDIA/NeMo-Guardrails/pull/759)) by @schuellc-nvidia
* Consistent uuid generation in debug mode ([#760](https://github.com/NVIDIA/NeMo-Guardrails/pull/760)) by @schuellc-nvidia
* Avatar posture management function in standard library ([#771](https://github.com/NVIDIA/NeMo-Guardrails/pull/771)) by @sklinglernv
* Nested ``if else`` construct parsing ([#833](https://github.com/NVIDIA/NeMo-Guardrails/pull/833)) by @radinshayanfar
* Multiline string values in interaction history prompting ([#765](https://github.com/NVIDIA/NeMo-Guardrails/pull/765)) by @radinshayanfar

## [2.0-beta.4] - 2024-10-02

### Fixed

* LLM prompt template ``generate_value_from_instruction`` for GPT and LLama model chat interface ([#775](https://github.com/NVIDIA/NeMo-Guardrails/pull/775)) by @schuellc-nvidia

## [2.0-beta.3] - 2024-09-27

### Added

* Support for new Colang 2 keyword `deactivate` ([#673](https://github.com/NVIDIA/NeMo-Guardrails/pull/673)) by @schuellc-nvidia
* Bot configuration as variable `$system.config` ([#703](https://github.com/NVIDIA/NeMo-Guardrails/pull/703)) by @schuellc-nvidia
* Basic support for most OpenAI and LLame 3 models ([#709](https://github.com/NVIDIA/NeMo-Guardrails/pull/709)) by @schuellc-nvidia
* Interaction loop priority levels for flows ([#712](https://github.com/NVIDIA/NeMo-Guardrails/pull/712)) by @schuellc-nvidia
* CLI chat debugging commands ([#717](https://github.com/NVIDIA/NeMo-Guardrails/pull/717)) by @schuellc-nvidia

### Changed

* Merged (and removed) utils library file with core library ([#669](https://github.com/NVIDIA/NeMo-Guardrails/pull/669)) by @schuellc-nvidia

### Fixed

* Fixes a event group match bug (e.g. `match $flow_ref.Finished() or $flow_ref.Failed()`) ([#672](https://github.com/NVIDIA/NeMo-Guardrails/pull/672)) by @schuellc-nvidia
* Fix issues with ActionUpdated events and user utterance action extraction ([#699](https://github.com/NVIDIA/NeMo-Guardrails/pull/699)) by @schuellc-nvidia

## [2.0-beta.2] - 2024-07-25

This second beta version of Colang brings a set of improvements and fixes.

### Added

Language and runtime:

* Colang 2.0 syntax error details ([#504](https://github.com/NVIDIA/NeMo-Guardrails/pull/504)) by @rgstephens
* Expose global variables in prompting templates ([#533](https://github.com/NVIDIA/NeMo-Guardrails/pull/533)) by @schuellc-nvidia
* `continuation on unhandled user utterance` flow to the standard library (`llm.co`) ([#534](https://github.com/NVIDIA/NeMo-Guardrails/pull/534)) by @schuellc-nvidia
* Support for NLD intents ([#554](https://github.com/NVIDIA/NeMo-Guardrails/pull/554)) by @schuellc-nvidia
* Support for the `@active` decorator which activates flows automatically ([#559](https://github.com/NVIDIA/NeMo-Guardrails/pull/559)) by @schuellc-nvidia

Other:

* Unit tests for runtime exception handling in flows ([#591](https://github.com/NVIDIA/NeMo-Guardrails/pull/591)) by @schuellc-nvidia

### Changed

* Make `if` / `while` / `when` statements compatible with python syntax, i.e., allow `:` at the end of line ([#576](https://github.com/NVIDIA/NeMo-Guardrails/pull/576)) by @schuellc-nvidia
* Allow `not`, `in`, `is` in generated flow names ([#596](https://github.com/NVIDIA/NeMo-Guardrails/pull/596)) by @schuellc-nvidia
* Improve bot action generation ([#578](https://github.com/NVIDIA/NeMo-Guardrails/pull/578)) by @schuellc-nvidia
* Add more information to Colang syntax errors ([#594](https://github.com/NVIDIA/NeMo-Guardrails/pull/594)) by @schuellc-nvidia
* Runtime processing loop also consumes generated events before completion ([#599](https://github.com/NVIDIA/NeMo-Guardrails/pull/599)) by @schuellc-nvidia
* LLM prompting improvements targeting `gpt-4o` ([#540](https://github.com/NVIDIA/NeMo-Guardrails/pull/540)) by @schuellc-nvidia

### Fixed

* Fix string expression double braces ([#525](https://github.com/NVIDIA/NeMo-Guardrails/pull/525)) by @schuellc-nvidia
* Fix Colang 2 flow activation ([#531](https://github.com/NVIDIA/NeMo-Guardrails/pull/531)) by @schuellc-nvidia
* Remove unnecessary print statements in runtime ([#577](https://github.com/NVIDIA/NeMo-Guardrails/pull/577)) by @schuellc-nvidia
* Fix `match` statement issue ([#593](https://github.com/NVIDIA/NeMo-Guardrails/pull/593)) by @schuellc-nvidia
* Fix multiline string expressions issue ([#579](https://github.com/NVIDIA/NeMo-Guardrails/pull/579)) by @schuellc-nvidia
* Fix tracking user talking state issue ([#604](https://github.com/NVIDIA/NeMo-Guardrails/pull/604)) by @schuellc-nvidia
* Fix issue related to a race condition ([#598](https://github.com/NVIDIA/NeMo-Guardrails/pull/598)) by @schuellc-nvidia

## [2.0-beta] - 2024-05-08

### Added

* [Standard library of flows](https://docs.nvidia.com/nemo/guardrails/colang-2/language-reference/the-standard-library.html): `core.co`, `llm.co`, `guardrails.co`, `avatars.co`, `timing.co`, `utils.co`.

### Changed

* Syntax changes:
  * Meta comments have been replaced by the `@meta` and `@loop` decorators:
    * `# meta: user intent` -> `@meta(user_intent=True)` (also user_action, bot_intent, bot_action)
    * `# meta: exclude from llm` -> `@meta(exclude_from_llm=True)`
    * `# meta: loop_id=<loop_id>`  -> `@loop("<loop_id>")`
  * `orwhen` -> `or when`
  * NLD instructions `"""<NLD>"""` -> `..."<NLD>"`
  * Support for `import` statement
  * Regular expressions syntax change `r"<regex>"` -> `regex("<regex>")`
  * String expressions change: `"{{<expression>}}"` -> `"{<expression>}"`

* Chat CLI runtime flags `--verbose` logging format improvements
* Internal event parameter renaming: `flow_start_uid` -> `flow_instance_uid`
* Colang function name changes: `findall` -> `find_all` ,

* Changes to flow names that were previously part of `ccl_*.co` files (which are now part of the standard library):
  * `catch colang errors` -> `notification of colang errors` (core.co)
  * `catch undefined flows` -> `notification of undefined flow start` (core.co)
  * `catch unexpected user utterance` -> `notification of unexpected user utterance` (core.co)
  * `poll llm request response` -> `polling llm request response` (llm.co)
  * `trigger user intent for unhandled user utterance` -> `generating user intent for unhandled user utterance` (llm.co)
  * `generate then continue interaction` -> `llm continue interaction` (llm.co)
  * `track bot talking state` -> `tracking bot talking state` (core.co)
  * `track user talking state` -> `tracking user talking state` (core.co)
  * `track unhandled user intent state` -> `tracking unhandled user intent state` (llm.co)
  * `track visual choice selection state` -> `track visual choice selection state` (avatars.co)
  * `track user utterance state` -> `tracking user talking state` (core.co)
  * `track bot utterance state` -> No replacement yet (copy to your bot script)
  * `interruption handling bot talking` -> `handling bot talking interruption` (avatars.co)
  * `generate then continue interaction` -> `llm continue interaction` (llm.co)

## [2.0-alpha] - 2024-02-28

[Colang 2.0](https://docs.nvidia.com/nemo/guardrails/colang-2/overview.html) represents a complete overhaul of both the language and runtime. Key enhancements include:

### Added

* A more powerful flows engine supporting multiple parallel flows and advanced pattern matching over the stream of events.
* A standard library to simplify bot development.
* Smaller set of core abstractions: flows, events, and actions.
* Explicit entry point through the main flow and explicit activation of flows.
* Asynchronous actions execution.
* Adoption of terminology and syntax akin to Python to reduce the learning curve for new developers.
