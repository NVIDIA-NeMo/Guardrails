// `Strings::DigitCount(text) -> { count: Long }`.
//
// Counts digit characters via the `regex_count` host function. In the
// policy the integer output is used inside arithmetic (`count + 1 <= 3`),
// which only the unwrapped form allows.
fn evaluate(text) {
    // Defensive per the provider contract: a provider may be evaluated
    // for ANY decision event, so any argument may be absent (unit).
    // Return a conforming sentinel instead of erroring (errors are UB).
    if type_of(text) == "()" {
        return #{ count: -1 };
    }

    #{ count: regex_count("[0-9]", text) }
}
