// `Regex::Analyze(text, pattern) -> { is_match, first_match, count }`.
//
// One provider that exercises ALL THREE regex host functions Dogwood
// exposes to provider scripts:
//   * regex_is_match(pattern, text) -> bool   — does the pattern match?
//   * regex_find(pattern, text)     -> string — the first match ("" if none)
//   * regex_count(pattern, text)    -> i64     — number of matches
//
// The output record surfaces each result so the policy can project and
// compare them independently.
fn evaluate(text, pattern) {
    // Defensive per the provider contract: a provider may be evaluated
    // for ANY decision event, so any argument may be absent (unit).
    // Return a conforming sentinel instead of erroring (errors are UB).
    if type_of(text) == "()" || type_of(pattern) == "()" {
        return #{ is_match: false, first_match: "", count: -1 };
    }

    #{
        is_match:    regex_is_match(pattern, text),
        first_match: regex_find(pattern, text),
        count:       regex_count(pattern, text),
    }
}
