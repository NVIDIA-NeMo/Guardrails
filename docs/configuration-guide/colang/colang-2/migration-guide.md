# Migrating from Colang 1 to Colang 2

The NeMo Guardrails CLI provides a tool (`nemoguardrails convert ...`) for converting guardrail configurations from Colang 1.0 format to Colang 2.x.
It performs several syntax transformations on the content of the files, such as converting certain keywords and changing the case of certain identifiers.

```{warning}
There are edge cases not convered by the conversion tool. You should always manually review a guardrail configuration after conversion.
```

## How it Works

The tool walks through the directory specified by the `path` argument and applies the transformations to each file ending with `.co`. The transformations include:

- Converting `define flow` to `flow`.
- Converting `define subflow` to `flow`.
- Converting `define bot` to `flow bot`.
- Converting `define user` to `flow user`.
- Converting `execute` to `await`.
- Converting snake_case identifiers after `await` to CamelCase and append it with key word `Action` while preserving any arguments used..
- Converting `else when` to `or when`.
- Converting `stop` to `abort`.
- Converting quoted strings after `bot` to `bot say` or `or bot say`.
- Converting quoted strings after `user` to `user said` or `or user said`.
- Convert anonymous flows.
- Use `active` decorator for root flows (flows that need activation).
- Add `global` keyword to global variables.
- Converting `...` to corresponding syntax in Colang 2.

### Generation operator

The syntax for the generation operation has changed slightly between Colang 1.0 and 2.0. The following Colang 1.0 snippet:

```colang
# some instruction in natural language
$var_name = ...
```

It is translated to:

```colang
$name = ... "some instruction in natural language"
```

### Generic Matching

In Colang 1.0 it was possible to use generic matching using `user ...` and `bot ...`.
These have a different equivalent in Colang 2.0. The `...` can no longer be used for matching, only for generation.

The following changes are applied during conversion:
- `user ...` is translated to `user said something`
- `bot ...` is translated to `bot said something`

### Rails Configuration

Colang 1.0 configuration can define certain rails in the `rails` field.
- If rails are defined in `config.yml` a `_rails.co` file is generated with the rails defined in it.

### Example Flow conversion

As an example, the following flow:

```colang
define flow
  user express greeting
  bot express greeting
```

is converted to:

```colang
@active
flow express_greeting
  user express greeting
  bot express greeting
```

```{note}
It is a convention to use past tense for user flows and present tense for bot flows. However, the migration tool does not enforce this convention.
```

The tool keeps track of the number of lines processed and the number of changes made in each file. It also counts the total number of files changed.

```{warning}
The tool modifies the original files. It is recommended to use version control to track the changes made by the tool. It also enables you to see the differences between the original and modified files.
```

## Usage

To use the conversion tool, use the following command:

```bash
nemoguardrails convert --from-version '1.0' "path/to/config"
```

The `convert` command has several options:

- `--from-version`: The version of the colang files to migrate from. Available options: ['1.0', '2.0-alpha']. Default is `1.0`.
- `--verbose` or `--no-verbose`: If the migration should be verbose and output detailed logs. Default is `no-verbose`.
- `--validate` or `--no-validate`: If the migration should validate the output using Colang Parser. Default is `no-validate`.
- `--use-active-decorator` or `--no-use-active-decorator`: If the migration should use the `active` decorator. Default is `use-active-decorator`.
- `--include-main-flow` or `--no-include-main-flow`: If the migration should add a main flow to the config. Default is `include-main-flow`.

## Assumptions and Limitations

- The tool assumes that the input files are correctly formatted. If a file is not correctly formatted, the tool may not work as expected.
- The tool uses regular expressions to find and replace certain patterns in the text. If the input files contain text that matches these patterns but should not be replaced, the tool may produce incorrect results.
- The tool renames the original files and writes the transformed content to new files with the original names. Use version control to track the changes made by the tool.
- The tool does not handle errors that may occur during file reading and writing operations. If an error occurs, the tool logs the error and continues with the next file.
- Using characters like `-`, `+` and tokens like `or`, `and`, etc. is not supported in flow definition, the migration tool does not handle this conversion.
- It is a better practice to define global variables at the beginning of the flow. However, the migration tool does not enforce this.
