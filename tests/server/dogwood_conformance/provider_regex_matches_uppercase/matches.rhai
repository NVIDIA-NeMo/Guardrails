// Information-provider implementation for `Strings::Matches`.
//
// A provider script defines `fn evaluate(arg0, arg1, …)` whose parameters
// correspond positionally to the declared `argumentTypes` in
// providers.json — here `(text, pattern)`. It returns a record whose
// shape matches the declared `outputType` (`{ matched: Bool }`).
//
// The engine runs this in a locked-down Rhai engine: no file/network/
// process access, only the host functions Dogwood registers. Here we use
// `regex_is_match(pattern, text) -> bool`.
fn evaluate(text, pattern) {
    // Defensive per the provider contract: a provider may be evaluated
    // for ANY decision event, so any argument may be absent (unit).
    // Return a conforming sentinel instead of erroring (errors are UB).
    if type_of(text) == "()" || type_of(pattern) == "()" {
        return #{ matched: false };
    }

    #{ matched: regex_is_match(pattern, text) }
}
